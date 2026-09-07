import os
import threading
from functools import lru_cache

import openai
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.base_agent import Agent

load_dotenv(override=True)

# API calling
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_API_KEY = os.getenv("GROQ_API_KEY")

# system_prompt acts here as structured data extractor(cleaning and standardizing).
SYSTEM_PROMPT = """You are a product data extractor. Summarize the user's product text.
Format your response exactly like this, replacing the bracketed placeholders with the actual product data:
Title: [Extract product title]
Category: [Determine category]
Brand: [Extract brand]
Description: [1 sentence description]
Details: [1 sentence on features]"""

class PreprocessorAgent(Agent):
    name = "Preprocessor Agent"
    color = Agent.WHITE

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
        daily_api_limit=800 # hard limit to prevent hitting daily quota
    ):
    
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # Metrics
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Rate Limiting & Safety Controls
        self.daily_api_limit = daily_api_limit
        self.api_calls_made = 0
        self._metrics_lock = threading.Lock()

    def messages_for(self, text: str) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    # Handles Groq's API limit
    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=3, max=65),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError))
    )
    def _call_api(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages_for(text),
            temperature=0.0, #Deterministic outputs are faster
            max_tokens=150  #Prevent runaway generation & cut latency 
        )

        # Thread-safe metrics update
        with self._metrics_lock:
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content.strip()

    # In-Memory caching: Repeated identical queries resolve in ~0.0ms and cost $0
    @lru_cache(maxsize=2048) # noqa: B019
    def _cached_preprocess(self, text: str) -> str:
        """
        Cached logic layer with graceful fallback and strict daily limits.
        """

        # Latency optimization: Skip LLM call entirely if text is already tiny
        if len(text.split()) < 5:
            return text

        # CIRCUIT BREAKER: Stop hitting the API if we process too many unique items in this run
        with self._metrics_lock:
            if self.api_calls_made >= self.daily_api_limit:
                print(f"⚠️ API Limit Guard: {self.api_calls_made} calls reached. Bypassing LLM for '{text[:20]}...'")
                return text
            self.api_calls_made += 1 

        try:
            return self._call_api(text)
        except Exception as e: # noqa : BLE001
            # Graceful Degradation: If API is down or hard rate-limited, fallback to raw text
            print(f"⚠️ Preprocessor API exhausted all retries: {e}. Falling back to raw text.")
            return text

    def preprocess(self, text) -> str:
        """
        Public entry point.
        Strips whitespace and standardizes casing to maximize cache hits.
        """
        candidate_keys = ["title", "product_description", "name", "description", "original_summary"]
        # safe_text: str =  str(text) if isinstance(text, str) else "Unknown Item"
        safe_text = str(text) if isinstance(text, str) else str(text.__dict__)
      
        if not isinstance(text, str):
            for key in candidate_keys:
                val = getattr(text, key, None)
                if val is None and isinstance(text, dict):
                    val = text.get(key)
                if val and str(val).strip():
                    safe_text = str(val).strip()
                    break

        clean_text = safe_text.strip()
        if not clean_text or clean_text == "Unknown Item":
            return "Unknown Product"
        
        return self._cached_preprocess(clean_text)