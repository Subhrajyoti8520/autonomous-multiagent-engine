import os
import re

# from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from bs4 import BeautifulSoup
from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.base_agent import Agent
from src.core.schemas import DealSelection, ScrapedDeal

FEEDS = feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

class ScannerAgent(Agent):
    name = "Scanner Agent"
    color = Agent.CYAN

    MODEL = "gemini-3.1-flash-lite"

    SYSTEM_PROMPT = """You are a precise data extraction system. Review the provided list of raw scraped deals.
    Select exactly the 5 deals with the most detailed hardware/product specifications and the most definitive final price.
    Output valid JSON matching the requested schema. Never output deals with ambiguous prices (e.g., '$50 off')."""
    
    USER_PROMPT_PREFIX = "Select the 5 most detailed deals with a clear price > 0 from the following data:\n\n"

    def __init__(self):
        super().__init__()
        self.log("Initializing Scanner Agent...")
        self.openai = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    @staticmethod
    def clean_html(html_snippet: str) -> str:
        """Helper to sanitize raw HTML from RSS summaries."""
        soup = BeautifulSoup(html_snippet, "html.parser")
        snippet = soup.find("div", class_="snippet summary")
        text = snippet.get_text(strip=True) if snippet else html_snippet
        text = re.sub(r"<[^<]+?>", "", text).replace("\n", " ")
        return text.strip()

    # def _scrape_single_url(self, entry: dict) -> Optional[ScrapedDeal]:
    def _scrape_single_url(self, entry: dict) -> ScrapedDeal | None:
        """Fetches and parses a single deal page with strict timeouts."""
        title = entry.get("title", "Unknown")
        url = entry["links"][0]["href"]
        summary = self.clean_html(entry.get("summary", ""))

        details, features = summary, ""

        try:
            # 8 second timeout prevents hanging threads on dead websites
            resp = requests.get(url, headers=self.headers, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                content_div = soup.find("div", class_="content-section")

                if content_div:
                    content = content_div.get_text().replace("\nmore", "").replace("\n", " ")
                    if "Features" in content:
                        parts = content.split("Features", 1)
                        details = parts[0][:600] # cap length to save LLM tokens
                        features = parts[1][:400] if len(parts) > 1 else ""
                    else:
                        details = content[:800]
        except requests.RequestException:
            self.log(f"Warning: HTTP fetch failed for {url}. Falling back to RSS summary.")

        return ScrapedDeal(title=title[:100], url=url, details=details, features=features)

    def fetch_deals(self, known_urls: set) -> list[ScrapedDeal]:
        """Concurrently scrapes deals from all RSS feeds."""
        raw_entries = []
        for feed_url in FEEDS:
            feed = feedparser.parse(feed_url)
            if hasattr(feed, 'entries'):
                # Grab top 10 per feed, filter out already seen URLs immediately
                for entry in feed.entries[:10]:
                    if entry["links"][0]["href"] not in known_urls:
                        raw_entries.append(entry)

        scraped_deals = []
        # Concurrently: Scrape upto 10 sites simultaneously
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_entry = {executor.submit(self._scrape_single_url, e): e for e in raw_entries}
            for future in as_completed(future_to_entry):
                deal = future.result()
                if deal:
                    scraped_deals.append(deal)

        self.log(f"Concurrently scraped {len(scraped_deals)} new deals.")
        return scraped_deals

    # Network resilience for Gemini API calls
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=2, max=20),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError))
    )
    def scan(self, memory: list[str] | None = None) -> DealSelection | None:
        """Analyzes scraped deals using Gemini Structured Outputs."""
        if memory is None:
            memory = []
        
        known_urls = set(memory) # O(1) lookup
        scraped = self.fetch_deals(known_urls)
        
        if not scraped:
            self.log("No new deals found to scan.")
            return None

        # Guardrail: Limit input to top 25 deals to prevent breaking Gemini's context window
        scraped = scraped[:25] 
        user_prompt = self.USER_PROMPT_PREFIX + "\n\n".join([d.describe() for d in scraped])

        self.log("Requesting structured DealSelection from Gemini...")
        result = self.openai.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=DealSelection,
            temperature=0.0 # Deterministic structured output
        )
        
        parsed_result = result.choices[0].message.parsed
        # Final validation
        parsed_result.deals = [d for d in parsed_result.deals if d.price > 0]
        
        self.log(f"Successfully extracted {len(parsed_result.deals)} deals.")
        return parsed_result
         