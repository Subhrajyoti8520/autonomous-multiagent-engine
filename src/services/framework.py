import json
import logging
import os
import sys
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from sklearn.manifold import TSNE

from src.agents.deterministic_planning_agent import DeterministicPlanningAgent
from src.core.schemas import Opportunity

load_dotenv(override=True)

import warnings

# Suppress the torchaudio backend dispatch UserWarning triggered by Streamlit's module watcher
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Torchaudio's I/O functions now support.*"
)

# Project root path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = str(ROOT_DIR / "data" / "vectorstore")
DEFAULT_MEMORY_PATH = str(ROOT_DIR / "data" / "memory.json")

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

CATEGORIES = [
    "Appliances", "Automotive", "Cell_Phones_and_Accessories",
    "Electronics", "Musical_Instruments", "Office_Products",
    "Tools_and_Home_Improvement", "Toys_and_Games"
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]

def init_logging():
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

class DealAgentFramework:
    def __init__(self, model_url: str | None = None, api_key: str | None = None, db_path: str = DEFAULT_DB_PATH, memory_path: str = DEFAULT_MEMORY_PATH):
        init_logging()
        self.db_path = db_path
        self.memory_filename = memory_path
        
        # Ensure data folder exists
        os.makedirs(os.path.dirname(self.memory_filename), exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection("products")
        self.memory = self.read_memory()
        self.model_url = model_url
        self.api_key = api_key
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework...")
            self.planner = DeterministicPlanningAgent(
                collection=self.collection,
                model_url=self.model_url
            )
            self.log("Agent Framework ready.")

    def read_memory(self) -> list[Opportunity]:
        if os.path.exists(self.memory_filename):
            try:
                with open(self.memory_filename, "r") as file:
                    data = json.load(file)
                return [Opportunity(**item) for item in data]
            except Exception as e: # noqa: BLE001
                self.log(f"Warning: Failed to load memory file ({e}). Starting fresh.")
        return []

    def write_memory(self) -> None:
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(self.memory_filename, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text) # noqa: LOG015

    def run(self) -> list[Opportunity]:
        self.init_agents_as_needed()
        self.log("Kicking off Planning Agent run...")
        
        # Extract URLs of previous deals so scanner avoids duplicates
        known_urls = [opp.deal.url for opp in self.memory if hasattr(opp, "deal")]
        result = self.planner.plan(memory=known_urls)
        
        if result:
            self.log(f"Deal accepted: {result.deal.product_description[:40]}... (Discount: ${result.discount:.2f})")
            self.memory.append(result)
            self.write_memory()
        else:
            self.log("No deal met the threshold this run.")
            
        return self.memory

    def get_plot_data(self, max_datapoints=2000):
        result = self.collection.get(include=["embeddings", "documents", "metadatas"], limit=max_datapoints)

        embeddings = result.get("embeddings")

        if embeddings is None:
            return [], np.array((0, 3)), []

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.size == 0:
            return [], np.empty((0, 3)), []

        if vectors.ndim != 2:
            return [], np.empty((0, 3)), []

        valid_mask = np.all(np.isfinite(vectors), axis=1)
        vectors = vectors[valid_mask]

        documents_raw = result.get("documents") or []
        # metadatas_raw = result.get("metadatas") or []

        documents = [doc for doc, valid in zip(documents_raw, valid_mask) if valid]
        # metadatas = [metadata or {} for metadata, valid in zip(metadatas_raw, valid_mask) if valid]
        n_samples = vectors.shape[0]

        # t-SNE requires more samples than components (3)
        if n_samples < 4:
            return documents, np.empty((0, 3)), []

        # Dynamically scale perplexity (must be less than n_samples)
        safe_perplexity = min(30, max(2, n_samples - 1))
        
        tsne = TSNE(n_components=3, random_state=42, perplexity=safe_perplexity, init="random")
        reduced_vectors = tsne.fit_transform(vectors)

        categories = [metadata.get("category", "Electronics") for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] if c in CATEGORIES else "gray" for c in categories]
        
        return documents, reduced_vectors, colors