from typing import Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

class Item(BaseModel):
    """Data-point for training and evaluating pricing models."""
    title: str
    category: str
    price: float
    full: str | None = None
    weight: float | None = None
    summary: str | None = None
    prompt: str | None = None
    id: int | None = None
    # full: Optional[str] = None
    # weight: Optional[float] = None
    # summary: Optional[str] = None
    # prompt: Optional[str] = None
    # id: Optional[int] = None
    

    def make_prompt(self, text: str):
        """Constructs the full training prompt with the target price."""
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """Truncates the prompt at PREFIX to safely pass to models during evaluation."""
        if not self.prompt:
            raise ValueError("Must call make_prompt() before generating a test_prompt.")
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push train/val/test splits to HuggingFace Hub."""
        DatasetDict({
            "train": Dataset.from_list([item.model_dump() for item in train]),
            "validation": Dataset.from_list([item.model_dump() for item in val]),
            "test": Dataset.from_list([item.model_dump() for item in test]),
        }).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load datasets from HuggingFace Hub and reconstruct Item objects."""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )