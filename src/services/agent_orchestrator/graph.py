"""LangGraph compilation: START → orchestrator → {fanout | responder} → END."""


def get_deep_agent():
    """Compile and return the deep agent LangGraph (lazy import)."""
    from src.services.agent_orchestrator.agent_factory import get_deep_agent as _get_deep_agent

    return _get_deep_agent()
