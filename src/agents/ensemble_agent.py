import concurrent.futures

from src.agents.base_agent import Agent
from src.agents.frontier_agent import FrontierAgent
from src.agents.neural_network_agent import NeuralNetworkAgent
from src.agents.preprocessor_agent import PreprocessorAgent
from src.agents.specialist_agent import SpecialistAgent


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(
        self, 
        collection = None, 
        api_key: str | None = None,
        specialist: SpecialistAgent | None = None,
        frontier: FrontierAgent | None = None,
        neural_network: NeuralNetworkAgent | None = None,
        preprocessor: PreprocessorAgent | None = None,
    ):
        """
        Accepts pre-initialized agent instances (Dependency Injection) 
        or creates default instances if not provided
        """
        super().__init__()
        self.log("Initializing Ensemble Agent")

        # SPECIALIST AGENT
        if specialist is not None:
            self.specialist = specialist
        else:
            raise ValueError("Must provide either an initialized 'specialist' agent or 'model_url'.")
            
        # FRONTIER AGENT (RAG)
        if frontier is not None:
            self.frontier = frontier
        elif collection is not None:
            self.frontier = FrontierAgent(collection=collection, api_key=api_key)
        else:
            raise ValueError("Must provide either an initialized 'frontier' agent or 'collection'.")
        
        # DEEP NEURAL NETWORK AGENT
        self.neural_network = neural_network if neural_network is not None else NeuralNetworkAgent()

        # PREPROCESSOR
        self.preprocessor = preprocessor if preprocessor is not None else PreprocessorAgent()

        self.log("Ensemble Agent is ready")


    def price(self, description: str, deal_id: str = "unknown") -> float:
        """
        Runs the full ensemble pipeline:
        1. Preprocesses raw unstructured web data.
        2. Queries Specialist, Frontier (RAG), and Neural Network sub-agents.
        3. Computes weighted linear ensemble price (60% Frontier, 20% Specialist, 20% Neural Network).
        """
        self.log("Running Ensemble Agent... ")
        
        rewrite = description
        
        self.log("Running Ensemble agent on preprocessed text")

        self.log(f"[{deal_id}]Dispatching child agents concurrently...") # Multi-Threading: Concurrency (Parallel Exectution)
        # Run all three pricing models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_specialist = executor.submit(self.specialist.price, rewrite)
            future_frontier = executor.submit(self.frontier.price, rewrite)
            future_neural_network = executor.submit(self.neural_network.price, rewrite)

            # Retrieve results as they finish
            specialist_val = future_specialist.result()
            frontier_val = future_frontier.result()
            neural_network_val = future_neural_network.result()

        # Dynamic weight redistribution (Fault Tolerance)
        # If an agent fails and returns 0.0, we redistribute its weight so it doesn't drag the price down.
        target_weights = {
            "frontier": 0.50,
            "specialist": 0.25,
            "neural_network": 0.25
        }

        actual_vals = {
            "frontier": frontier_val,
            "specialist": specialist_val,
            "neural_network": neural_network_val
        }

        total_valid_weight = 0.0
        combined = 0.0

        for key, val in actual_vals.items():
            if val > 0.0: # Agent succeeded
                total_valid_weight += target_weights[key]
                combined += val * target_weights[key]

        if total_valid_weight > 0:
            combined = combined / total_valid_weight # Normalize based on surviving agents
        else:
            combined = 0.0 # Total failure case

        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined
