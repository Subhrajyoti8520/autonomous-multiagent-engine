import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn  # noqa PLR0402
from sklearn.feature_extraction.text import HashingVectorizer

from src.agents.base_agent import Agent

# Dynamically find the project root:
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "deep_neural_network.pth"

class ResidualBlock(nn.Module): # ResNet-styls skip connections for dense layers
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x) # optimized skip connection

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)

class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    # Target variable normalization parameters
    Y_STD = 1.0328539609909058
    Y_MEAN = 4.434937953948975

    # Shared model prevents memory leaks if initialized multiple times
    _shared_model = None

    def __init__(self, weights_path: str = str(DEFAULT_WEIGHTS_PATH)):
        """
        Initialize the model, vectorizer, sets the compute device,
        and loads weights dynamically.
        """
        super().__init__()
        self.log("Initializing PyTorch Model & Vectorizer...")

        # Setup Vectorizer
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)

        # Setup Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if NeuralNetworkAgent._shared_model is None:
            self.log(f"Loading weights into {self.device} memory globally...")
            model = DeepNeuralNetwork(input_size=5000)

            # Force a crash if the weights are missing
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"CRITICAL: Weights file not found at {weights_path}. "
                    "Fix DEFAULT_WEIGHTS_PATH or pass the correct path."
                )
                
            model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.log(f"Successfully loaded weights from {weights_path}")

            model.to(self.device)
            model.eval()
            NeuralNetworkAgent._shared_model = model

        self.model = NeuralNetworkAgent._shared_model
        self.log("Neural Network Agent is ready.")
    
    def price(self, description: str) -> float:
        """
        Vectorizes text and runs fast inference to estimate price.
        """
        self.log("Starting prediction...")

        if hasattr(description, "summary"):
            safe_desc = description.summary or description.title
        elif isinstance(description, dict):
            safe_desc = description.get("summary", description.get("title", ""))
        else:
            safe_desc = str(description) if description else ""

        with torch.inference_mode(): #faster than no_grad()
            # Vectorize text(returns sparse matrix, convert to dense numpy float32)
            vector_np = self.vectorizer.transform([safe_desc]).toarray().astype(np.float32)

            # Optimized tensor conversion (avoids memory copy)
            vector_tensor = torch.from_numpy(vector_np).to(self.device)

            # Forward pass
            pred = self.model(vector_tensor)[0]

            # De-normalize output
            result = torch.exp(pred * self.Y_STD + self.Y_MEAN) - 1
            result = result.item()

        final_price = max(0.0, result)
        self.log(f"Prediction complete: ${final_price:.2f}")
        return final_price