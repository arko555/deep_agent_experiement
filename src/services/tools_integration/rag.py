import os
from typing import Literal

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> str:
    """Search the web for current information. Use when you need data from the internet.

    Args:
        query: The search query.
        max_results: The maximum number of results to return.
        topic: The search topic ('general', 'news', 'finance').

    Returns:
        A markdown formatted string of search results.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Tavily API key not found. Please set TAVILY_API_KEY in .env."
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=tavily_api_key)
        res = tavily.search(query, max_results=max_results, topic=topic)
        results = res.get("results", []) if isinstance(res, dict) else []
        if not results:
            return "No results found."
        lines = ["## Search results for: " + query, ""]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")
            lines.append(f"{i}. **{title}**")
            if url:
                lines.append(f"   URL: {url}")
            if content:
                # Truncate long content
                snippet = content[:300].replace("\n", " ")
                lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching the web: {str(e)}"
