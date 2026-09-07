from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.base_agent import Agent
from src.agents.ensemble_agent import EnsembleAgent
from src.agents.messaging_agent import MessagingAgent
from src.agents.preprocessor_agent import PreprocessorAgent
from src.agents.scanner_agent import ScannerAgent
from src.agents.specialist_agent import SpecialistAgent
from src.core.schemas import Deal, Opportunity


class DeterministicPlanningAgent(Agent):
    name = "Deterministic Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 5.0

    def __init__(self, collection, model_url: str):
        """
        Create instances of the Agents that this planner coordinates across
        """
        super().__init__()
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.preprocessor = PreprocessorAgent()

        self.specialist = SpecialistAgent(host=model_url)
        # Dependency Injection into EnsembleAgent 
        self.ensemble = EnsembleAgent(
            collection=collection,
            specialist=self.specialist,
            preprocessor=self.preprocessor
        )
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity: 
        """Isolated evaluation task for execution."""
        raw_text = (
            getattr(deal, "product_description", None) 
            or getattr(deal, "details", None) 
            or getattr(deal, "title", None) 
            or getattr(deal, "summary", "")
        )

        if not raw_text or not str(raw_text).strip():
            self.log(f"⚠️ WARNING: Deal price=${deal.price} has NO product description/title! Skipping evaluation.")
            return Opportunity(deal=deal, estimate=deal.price, discount=0.0)

        try:
            clean_text = self.preprocessor.preprocess(raw_text)
            if not clean_text or not clean_text.strip():
                clean_text = raw_text[:300]  # Fallback to raw text if preprocessor empties it
        except Exception as e: # noqa: BLE001
            self.log(f"Preprocessor failed ({e}), using raw text fallback.")
            clean_text = raw_text[:300]

        deal.product_description = clean_text

        self.log(f"Pricing deal [${deal.price}]: '{clean_text[:60]}...'")

        estimate = self.ensemble.price(clean_text, deal_id=deal.url) 
        discount = estimate - deal.price
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: list[str] | None = None) -> Opportunity | None:
        """Run the Full workflow. ScannerAgent -> PreprocessorAgent -> EnsembleAgent -> MessagingAgent"""
        if memory is None:
            memory = []

        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)

        if not selection or not selection.deals:
            self.log("Planning Agent found no deals to process.")
            return None

        # Strictly limit to top 5 deals to prevent long queue bottlenecks
        deals_to_evaluate = selection.deals[:5]
        self.log(
            f"Evaluating top {len(deals_to_evaluate)} deals sequentially..."
        )

        # Record URLs in memory so subsequent runs ignore these items
        for deal in deals_to_evaluate:
            if getattr(deal, "url", None):
                memory.append(deal.url)

        opportunities = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_deal = {executor.submit(self.run, deal): deal for deal in deals_to_evaluate}
            for future in as_completed(future_to_deal):
                try:
                    opportunities.append(future.result())
                except Exception as e: # noqa: BLE001
                    self.log(f"Evaluation failed for a deal: {e}")

        if not opportunities:
            return None

        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]

        self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
        if best.discount > self.DEAL_THRESHOLD:
            self.messenger.notify(
                best.deal.product_description, best.deal.price, best.estimate, best.deal.url
            )
            return best
        
        return None
