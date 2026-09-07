import os
import re

import requests

from src.agents.base_agent import Agent


class SpecialistAgent(Agent):

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self, model_name: str = "specialist-pricer", host: str | None = None):
        """
        Initializes the agent targeting the local Ollama API.
        """
        super().__init__()

        resolved_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        self.model_name = model_name
        self.endpoint = f"{resolved_host.rstrip('/')}/api/generate"

        # Accept only a standalone numeric price from the model.
        self.price_pattern = re.compile(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)")

        # Reuse HTTP connections to reduce local network latency.
        self.session = requests.Session()

        self.log(
            f"Specialist Agent initialized: "
            f"targeting local Ollama ({self.model_name})"
        )

    def _extract_description(self, description) -> str:
        """
        Extract a usable product description from either a string,
        dictionary, or object.
        """
        if isinstance(description, str):
            text = description.strip()
            return text if text else "Unknown Item"

        candidate_keys = [
            "title",
            "product_description",
            "name",
            "description",
            "product_name",
            "original_summary",
        ]

        # Handle dictionaries.
        if isinstance(description, dict):
            for key in candidate_keys:
                value = description.get(key)

                if value is not None and str(value).strip():
                    return str(value).strip()

        for key in candidate_keys:
            value = getattr(description, key, None)

            if value is not None and str(value).strip():
                return str(value).strip()

        return "Unknown Item"

    def _parse_price(self, response_text: str) -> float:
        """
        Parse a price only when Ollama returns a clean numeric value.
        We prefer an explicitly labelled price. Otherwise, because the
        prompt asks for a single number, we accept the first numeric value.
        """

        if not response_text:
            raise ValueError("Ollama returned an empty response")

        text = response_text.strip()
        text = text.replace(",", "").strip()

        dollar_match = re.search(r"\$\s*(\d*\.\d+|\d+)", text)
        if dollar_match:
            try:
                return max(0.0, float(dollar_match.group(1)))
            except ValueError:
                pass

        numbers = re.findall(r"\d*\.\d+|\d+", text)
        if numbers:
            try:
                price = float(numbers[0])
                return max(0.0, price)
            except ValueError:
                pass
        
        # If absolutely no numbers exist in the string
        self.log(f"Could not extract a valid price from response: {response_text!r}")
        return 0.0

    def price(self, description) -> float:
        """
        Predict the product's market price using the local specialist model.
        """
        safe_desc = self._extract_description(description)
        self.log(f"Sending product to Specialist model: {safe_desc[:120]!r}")

        prompt = f"What does this cost to the nearest dollar?\n\n{safe_desc}\n\nPrice is $"

        try:
            response = self.session.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 15,
                        "seed": 42,
                        "stop": ["\n", "<|im_end|>", "<|eot_id|>"],
                    },
                },
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()
            result_text = str(result.get("response", "")).strip()

            self.log(f"Specialist raw response: {result_text!r}")
            final_price = self._parse_price(result_text)
            self.log(f"Specialist Agent completed - predicted ${final_price:.2f}")
            return final_price

        except Exception as e: # noqa BLE001
            self.log(f"Specialist Agent error: {e}")
            return 0.0