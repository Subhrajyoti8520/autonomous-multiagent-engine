import json
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from src.agents.base_agent import Agent
from src.agents.ensemble_agent import EnsembleAgent
from src.agents.messaging_agent import MessagingAgent
from src.agents.preprocessor_agent import PreprocessorAgent
from src.agents.scanner_agent import ScannerAgent
from src.agents.specialist_agent import SpecialistAgent
from src.core.exceptions import (
    AgentExecutionError,
    ExtractionError,
    ModelInferenceError,
)
from src.core.schemas import Deal, Opportunity


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gemini-3.5-flash"

    def __init__(self, collection, model_url: str, api_key: str | None = None):
        """
        Create instances of the Agents that this planner coordinates across
        """
        super().__init__()
        self.log("Autonomous Planning Agent is initializing")

        # Sub-Agents
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

        self.openai = OpenAI(
            api_key = api_key or os.environ.get("GEMINI_API_KEY"),
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.memory = None 
        self.opportunity = None 
        self.log("Autonomous Planning Agent is ready")

    def _clean_deal(self, deal: Deal) -> Deal:
        """Helper to run the preprocessor via Groq"""
        deal.product_description = self.preprocessor.preprocess(deal.product_description)
        return deal

    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the tool to scan, then clean the text before returning to the LLM.
        """
        self.log("Autonomous Planning agent is calling scanner")
        try:
            results = self.scanner.scan(memory=self.memory)
        except Exception as e:
            self.log(f"⚠️ Scanner failure: {e}")
            raise ExtractionError(f"Deal extraction failed: {e}") from e

        if not results or not results.deals:
            return "No deals found"

        results.deals = results.deals[:5]

        self.log(f"Preprocessing {len(results.deals)} deals via Groq...")

        # Clean text concurrently before the LLM evaluates them
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(self._clean_deal, results.deals))

        return results.model_dump_json()

    def estimate_true_value(self, description: str) -> str:
        """
        Run the tool to estimate true value
        """
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        try:
            estimate = self.ensemble.price(description)
        except Exception as e:
            self.log(f"⚠️ Pricing consensus failed for '{description[:30]}...': {e}")
            raise ModelInferenceError(f"Consensus pricing failed: {e}") from e
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(
        self, description: str, deal_price: float, estimated_true_value: float, url: str
    ) -> dict:
        """
        Run the tool to notify the user
        """
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
        else:
            self.log("Autonomous Planning agent is notifying user")
            self.messenger.notify(description, deal_price, estimated_true_value, url)
            deal = Deal(product_description=description, price=deal_price, url=url)
            discount = estimated_true_value - deal_price
            self.opportunity = Opportunity(
                deal=deal, estimate=estimated_true_value, discount=discount
            )
        return "Notification sent ok"

    scan_function = { # noqa: RUF012
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = { # noqa: RUF012
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = { # noqa: RUF012
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item itself scraped from the internet",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal scraped from the internet",
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual value that this is worth",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of this deal as scraped from the internet",
                },
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        """
        Return the json for the tools to be used
        """
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        """
        Actually call the tools associated with this message, with JSON safety checks.
        """
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            # Robust JSON check to prevent crashes if LLM hallucinated the tool format
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                self.log(f"⚠️ JSON decode failed for {tool_name}. Passing empty args.")
                arguments = {}

            tool = mapping.get(tool_name)
            try:
                result = tool(**arguments) if tool else ""
            except (ModelInferenceError, ExtractionError) as e:
                self.log(f"⚠️ Domain error in {tool_name}: {e}")
                result = f"Error executing tool: {e}"
            except Exception as e:
                self.log(f"⚠️ Unexpected runtime crash in {tool_name}: {e}")
                raise AgentExecutionError(f"Fatal tool execution error: {e}") from e

            # Defensive conversion: Ensure result is a string for the LLM
            if not isinstance(result, str):
                result = json.dumps(result)

            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
    user_message = """
    First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """
    messages = [ # noqa: RUF012
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: list[str] | None = None) -> Opportunity | None:
        """
        Run the full workflow, providing the LLM with tools to surface scraped deals to the user
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        if memory is None:
            memory = []

        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:] #initializes the conversation history
        done = False

        # Add a safety counter
        loops = 0
        MAX_LOOPS = 10

        while not done and loops < MAX_LOOPS:
            loops += 1
            try:
                response = self.openai.chat.completions.create(
                    model=self.MODEL, messages=messages, tools=self.get_tools()
                )
            except Exception as e:
                raise AgentExecutionError(f"Planning LLM call failed: {e}") from e
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        
        if loops >= MAX_LOOPS:
            self.log("⚠️ Autonomous Planner hit max loops and was terminated early.")
            raise AgentExecutionError("Planner exceeded maximum iteration loops without resolving.")
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
