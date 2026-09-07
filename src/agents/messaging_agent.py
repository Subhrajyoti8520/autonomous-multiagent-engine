import os

import requests
from litellm import completion

from src.agents.base_agent import Agent
from src.core.schemas import Opportunity


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "groq/openai/gpt-oss-20b"
    # PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

    def __init__(self):
        """
        Set up this object to either do push notifications via Pushover.
        """
        super().__init__()
        self.log("Messaging Agent is initializing")
        self.ntfy_url = os.getenv("NTFY_URL", "https://ntfy.sh")
        self.ntfy_topic = os.getenv("NTFY_TOPIC")
        # self.pushover_user = os.getenv("PUSHOVER_USER")
        # self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.log("Messaging Agent has initialized Ntfy and Groq")

    def push(self, text: str, url: str | None = None):
        """
        Send a Push Notification using the NTFY API.
        """
        self.log("Messaging Agent is sending a push notification via ntfy")
        payload = {
            # "user": self.pushover_user,
            # "token": self.pushover_token,
            "topic": self.ntfy_topic,
            "message": text,
            "sound": "cashregister",
            "title": "💰 Price is Right AI: Deal Found!",
            "tags": ["moneybag", "robot"]
        }

        # Utilize Pushover's native URL fields for a cleaner mobile UX
        if url:
            # payload["url"] = url
            # payload["url_title"] = "View Deal Now"
            # Native ntfy clickable button action
            payload["actions"] = [{
                "action": "view",
                "label": "View Deal Now",
                "url": url,
                "clear": True
            }]

        try:
            # 5-second timeout prevents the agent from hanging on bad network connections
            response = requests.post(self.ntfy_url, json=payload, timeout=5)

            if response.status_code == 200:
                self.log("✅ Push notification sent successfully.")
            else:
                self.log("⚠️ Ntfy API Error {response.status_code}: {response.text}")

        except requests.RequestException as e:
            self.log(f"⚠️ Failed to deliver push notification: {e}")

    def alert(self, opportunity: Opportunity):
        """Template-based fallback alert (No LLM)"""
        text = f"Estimate: ${opportunity.estimate:.2f} | Price: ${opportunity.deal.price:.2f}\n"
        text += f"Total Discount: ${opportunity.discount:.2f}\n\n"
        text += f"{opportunity.deal.product_description[:80]}..."
        
        self.push(text, url=opportunity.deal.url)
        self.log("Messaging Agent has completed template alert")

    def craft_message(self, description: str, deal_price: float, estimated_true_value: float) -> str:
        """Uses LLM(Groq) to write an exciting summary."""
        user_prompt = (
            "Write a 2-sentence exciting push notification for this deal.\n"
            f"Item: {description}\n"
            f"Offered Price: ${deal_price:.2f}\n"
            f"True Value: ${estimated_true_value:.2f}\n\n"
            "Keep it under 150 characters total. Do not include URLs or hashtags."
        )
        
        response = completion(
            model=self.MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3, # Low temp keeps the excitement grounded, prevents weird formatting
            max_tokens=60
        )
        return response.choices[0].message.content.strip()    

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str):
        """
        Main entrypoint: Crafts LLM message and dispatches it.
        """
        self.log("Messaging Agent is using Groq to craft the message")
        try:
            text = self.craft_message(description, deal_price, estimated_true_value)
        except Exception as e: # noqa: BLE001
            self.log(f"Groq generation failed, falling back to basic text: {e}")
            text = f"Amazing deal! Only ${deal_price:.2f} (Worth ${estimated_true_value:.2f})."

        self.push(text, url=url)
        self.log("Messaging Agent has completed")
