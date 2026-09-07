from pydantic import BaseModel, Field, model_validator

# from typing import List, Optional

class Deal(BaseModel):
    """A standardized representation of an extracted deal."""

    title: str | None = Field(
        default=None,
        description="The short title or brand/model name of the product."
    )

    product_description: str = Field(
        default = "",
        description="A clear, 3-4 sentence summary focusing purely on the product's features and specifications. Do not include promotional terms, discount amounts, or coupons."
    )
    price: float = Field(
        description="The final checkout price of the product as a float. Must be greater than 0. If a deal is '$100 off $300', the price is 200.0."
    )
    url: str = Field(
        default = "",
        description="The source URL of the deal."
    )

    @model_validator(mode="before")
    @classmethod
    def sync_title_and_description(cls, values):
        if isinstance(values, dict):
            t = values.get("title")
            desc = values.get("product_description") or values.get("description") or values.get("rewritten_summary") or values.get("original_summary")
            
            # Cross-fill if either is missing
            if not t and desc:
                values["title"] = desc[:80].strip()
            if not values.get("product_description") and t:
                values["product_description"] = t

        return values

    @property
    def display_title(self) -> str:
        """Returns the title, product_description, or fallback string."""
        return self.title or self.product_description or "Unknown Item"

class DealSelection(BaseModel):
    """Structured output format for the Scanner LLM."""
    deals: list[Deal] = Field(
        description="Exactly 5 of the highest quality deals with clear specifications and highly confident prices."
    )

class Opportunity(BaseModel):
    """Represents a deal evaluated by the pricing agents as profitable."""
    deal: Deal
    estimate: float
    discount: float

class ScrapedDeal(BaseModel):
    """Internal model for raw scraped data before LLM processing."""
    title: str
    url: str
    details: str
    features: str

    def describe(self) -> str:
        return f"Title: {self.title}\nDetails: {self.details}\nFeatures: {self.features}\nURL: {self.url}"
        