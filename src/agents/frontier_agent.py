import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import openai
import torch
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.base_agent import Agent

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
VECTOR_DB_DIR = (str(ROOT_DIR / "data" / "vectorstore"))
MODEL_CACHE_DIR = str(ROOT_DIR / "data" / "models" / "transformers") 
CACHE_FILE = ROOT_DIR / "data" / "frontier_cache.json"

# # Prevent huggingface_hub from defaulting to ~/.cache/huggingface/hub
# os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)

class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.CYAN

    # Class-level variables to share across instances and prevent VRAM exhaustion
    _encoder = None
    _cross_encoder = None

    def __init__(self, collection=None, api_key: str | None = None):
        """
        Initializes the Frontier RAG Agent with Bi-Encoder for retrieval, 
        Cross-Encoder for reranking, and Gemini API for rewriting & synthesis.
        """
        super().__init__()
        self.log("Initializing Frontier Agent...")

        # Initialize Vector Database Collection
        if collection is None:
            db_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            self.collection = db_client.get_or_create_collection(name="products")
        else:
            self.collection = collection

        # Initialize API Client
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.log("WARNING: GEMINI_API_KEY is not set. Synthesis may fail.", level="WARNING")

        self.llm_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
            api_key=api_key
        )

        # Configure model 
        self.model_name = "gemini-3.1-flash-lite"
        self.instruction_prefix = "Represent this sentence for searching relevant passages: "

        # Improved regex to safely extract currency
        self.price_pattern = re.compile(r"\$?\s*(?:[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")

        # Persistent cache setup
        self.cache_file = Path(CACHE_FILE)
        self.cache: dict[str, float] = self._load_cache()

        # Dynamic device detection (Support for NVIDIA, Apple Silicon, and CPU)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Lazy load models globally in FP16 to save 50% VRAM
        if FrontierAgent._encoder is None:
            self.log(f"Loading BAAI Bi-Encoder globally on {device}...")
            FrontierAgent._encoder = SentenceTransformer(
                'BAAI/bge-large-en-v1.5', 
                device=device, 
                model_kwargs={
                    "torch_dtype": "auto", # Uses FP16 if supported
                    "cache_dir": MODEL_CACHE_DIR,
                }, 
            )
            
        if FrontierAgent._cross_encoder is None:
            self.log(f"Loading BAAI Cross-Encoder globally on {device}...")
            FrontierAgent._cross_encoder = CrossEncoder(
                'BAAI/bge-reranker-base', 
                device=device, 
                cache_folder=MODEL_CACHE_DIR,
                model_args={
                    "cache_dir": MODEL_CACHE_DIR,
                },
                activation_fn=torch.nn.Identity(), 
                # ^ Optional: speeds up if you don't need normalized scores
            )
        
        self.encoder = FrontierAgent._encoder
        self.cross_encoder = FrontierAgent._cross_encoder
        self.log("Frontier Agent initialized.")

    def _load_cache(self) -> dict[str, float]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e: # noqa: BLE001
                self.log(f"Cache load error: {e}. Starting fresh.")
        return {}

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(openai.RateLimitError)
    )
    def _call_llm_with_retry(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        """Helper method to handle Gemini API rate limits automatically."""
        completion = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()
    
    def _rewrite_query(self, raw_query: str) -> str:
        """
        Rewrites the query using LLM for optimal semantic search density.
        Uses a lightweight, low-latency prompt.
        """
        # Uncomment to use query rewriting by LLM
        # Latency Optimization: skip rewrite if the query is already concise
        if len(raw_query.split()) < 10:
            self.log("Query is short; skipping rewrite.")
            return raw_query

        self.log(f"Rewriting query: {raw_query[:50]}...")
        prompt = (
            "You are a search expert. Rewrite the following product description into a concise, "
            "highly specific search query to find similar products and their prices in a database. "
            "Output ONLY the raw search query string and nothing else.\n\n"
            f"Product: {raw_query}"
        )
        try:
            rewritten = self._call_llm_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for deterministic outputs
                max_tokens=50  
            ).strip("'\"")
            self.log(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e: # noqa: BLE001
            self.log(f"Query rewrite failed: {e}. Using raw query.")
            return raw_query
        # """
        # Bypassed to conserve API Requests-Per-Day limit. 
        # Returns the raw query directly for the Bi-Encoder.
        # """
        # self.log("Skipping LLM rewrite to save API quota. Using raw query.")
        # return raw_query

    def _retrieve_and_rerank(self, query: str, top_k: int = 5, rerank_k: int = 3) -> list[dict]:
        """
        1. Embeds the query using BAAI Bi-Encoder.
        2. Retrieves top_k candidates via ChromaDB.
        3. Reranks candidates using BAAI Cross-Encoder for ultimate precision.
        """
        # Embed query with BGE instruction prefix
        full_query = self.instruction_prefix + query
        query_embedding = self.encoder.encode(full_query, normalize_embeddings=True).tolist()

        db_count = self.collection.count()
        if db_count == 0:
            print("\n[!] WARNING: ChromaDB is empty! RAG has no context.")
            return []

        # use min() to prevent errors if the DB has fewer items that top_k(=5 to reduce cpu reranking latency)
        safe_top_k = min(top_k, db_count)
        if safe_top_k == 0:
            return []

        # Dense Retrieval (Fast, wide net)
        search_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k
        )

        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]

        if not documents:
            return []

        # Prepare pairs for Cross-Encoder(Query, Document)
        cross_input = [[query, doc] for doc in documents] 

        # Predict relevance scores (Slower, but highly accurate)
        rerank_scores = self.cross_encoder.predict(cross_input)

        # Prevent zip crash if cross_encoder returns a single float instead of a list
        if isinstance(rerank_scores, (float, np.float32)):
            rerank_scores = [rerank_scores]

        # Zip, sort and select top K
        results = []
        for doc, meta, score in zip(documents, metadatas, rerank_scores):
            results.append({"document": doc, "metadata": meta, "score": score})

        # Sort descending by cross-encoder score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:rerank_k]

    def price(self, description: Any) -> float:
        """
        Main pipeline execution:
        1. Rewrite query(if necessary).
        2. Retrieve and Rerank context.
        3. Synthesize final price estimation via LLM.
        """

        if isinstance(description, str):
            safe_desc = description.strip()
        elif hasattr(description, "product_description") and description.product_description:
            safe_desc = str(description.product_description).strip()
        elif hasattr(description, "title") and description.title:
            safe_desc = str(description.title).strip()
        elif isinstance(description, dict):
            safe_desc = str(description.get("product_description") or description.get("title") or "").strip()
        else:
            safe_desc = str(description).strip()

        if not safe_desc or safe_desc == "Unknown Item":
            return 0.0

        # CHECK CACHE FIRST
        cache_key = self._get_hash(safe_desc)
        if cache_key in self.cache:
            self.log("Cache hit! Returning cached price.")
            return self.cache[cache_key]

        self.log(f"Starting pricing pipeline for: {safe_desc[:30]}...")

        # Query Rewriting
        search_query = self._rewrite_query(safe_desc)

        # Advanced RAG(Retrieve + Rerank)
        reranked_results = self._retrieve_and_rerank(search_query, top_k=5, rerank_k=3)
        
        # Build Context String
        context_str = ""
        for i, res in enumerate(reranked_results, 1):
            doc = res['document']
            price = res['metadata'].get('price', 'Unknown')
            context_str += f"Reference {i}:\nDescription: {doc}\nPrice: ${price}\n\n"

        # Synthesis
        sys_prompt = (
            "You are an expert appraiser. Estimate the fair market price for the target item "
            "Heavily weight references that match the exact brand and model. If a reference is clearly "
            "a different product tier, ignore it. Output ONLY the final estimated price as a number "
            "(e.g., '149.99'). Do not explain your reasoning."
        )
        user_prompt = (
            f"References:\n{context_str}\n"
            f"Target Item:\n{safe_desc}\n\n"
            "Based on the references above, provide the final price estimate for the target item. "
            "Output ONLY the numeric value without currency symbols."
        )

        self.log("Synthesizing final price estimation...")
        try:
            result_text = self._call_llm_with_retry(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=50 
            )
            
            # Clean and extract price using the safer regex
            match = self.price_pattern.search(result_text.replace(",", ""))
            if match:
                clean_str = match.group().replace('$', '').strip()
                final_price = float(clean_str)

                # SAVE SUCCESSFUL PREDICTIONS TO CACHE
                if final_price > 0.0:
                    self.cache[cache_key] = final_price
                    self._save_cache()

            else:
                # Hard print to prevent silent failures if regex misses
                print(f"\n[?] Regex failed to extract price. Raw LLM Output: '{result_text}'")
                final_price = 0.0
            
            self.log(f"Frontier Agent completed - predicted ${final_price:.2f}")
            return final_price
            
        except Exception as e: # noqa: BLE001
            print(f"\n[!] LLM Synthesis Error: {e!r}")
            return 0.0