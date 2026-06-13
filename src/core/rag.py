import os
from typing import Literal
from tavily import TavilyClient

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
):
    """Search the web for current information. Use when you need data from the internet."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

    if not tavily:
        return "Tavily API key not found. Please set TAVILY_API_KEY in .env."
    return tavily.search(query, max_results=max_results, topic=topic)
