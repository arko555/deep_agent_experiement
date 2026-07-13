from typing import List
from pydantic import BaseModel, Field
from src.core.memory import get_memory_content

class ResearchResult(BaseModel):
    summary: str = Field(description="A comprehensive summary of the research findings.")
    key_findings: List[str] = Field(description="A list of key facts and findings discovered.")
    sources: List[str] = Field(description="A list of sources used for the research.")
    confidence_score: float = Field(description="A score from 0 to 1 representing the confidence in the findings.")

def get_researcher_system_prompt() -> str:
    """
    Generates the specialized system prompt for the Researcher agent.
    """
    prompt = """You are a specialized Research Subagent. Your sole objective is to conduct thorough, accurate, and comprehensive research on the requested topic.

Instructions:
1. Break down the research query into focused search queries.
2. Use the Tavily search tool (`internet_search`) to gather results.
3. If search results are insufficient, try alternative search queries.
4. Always prioritize factual accuracy, detailed source synthesis, and depth.
5. Save a detailed summary of your findings as a markdown file under the `./workspace` directory (e.g. `workspace/research_summary.md`).
6. Strictly do NOT write files outside `./workspace` or to `/tmp/`.

Follow any conventions listed in AGENTS.md below.

Your final response must be a structured object containing:
- summary: A comprehensive summary of the research findings.
- key_findings: A list of key facts and findings discovered.
- sources: A list of sources used for the research.
- confidence_score: A score from 0 to 1 representing the confidence in the findings.
"""
    agents_md = get_memory_content()
    if agents_md:
        prompt += f"\n\n=== Shared Data / AGENTS.md ===\n{agents_md}"
    return prompt
