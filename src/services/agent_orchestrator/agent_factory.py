"""Deep agent graph factory.

Constructs and caches the LangGraph state graph for the deep agent workflow.

This module used to own the top-level ``tools`` node, where the orchestrator
called tools itself and delegated to sub-agents via the ``task`` tool. Both are
gone: the orchestrator is a pure router, tool calls happen inside a
sub-agent's ReAct loop, and the graph itself is assembled in
``graph.get_deep_agent``. What remains here is model selection, the
observability tracer, and the cache-reset hooks.
"""

import os
import logging
from datetime import datetime

from dotenv import load_dotenv

# OpenRouter-only for now: the other provider clients are imported by the
# commented fallback chain in _get_cached_model below. Re-import when
# restoring that chain.
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain_core.callbacks import BaseCallbackHandler

from src.services.tools_integration.mcp_client import clear_mcp_tools_cache as clear_mcp_client_cache
from src.services.tools_integration.mcp_bridge import clear_mcp_tools_cache as clear_mcp_bridge_cache
from src.services.tools_integration.guardrails import get_workspace_root

load_dotenv()

logger = logging.getLogger(__name__)


# --- Cached Compiled Graph ---
# Compile once, reuse across all invocations (including recursive subagents).
# Safe because node lambdas call get_model() at runtime, not compile time.
_compiled_graph = None


# --- Observability Hooks (4.4, made real in 8.3) ---
# Real LangChain callback handler for monitoring LLM calls, tool executions,
# and chain invocations. Enable via OBSERVABILITY=1 env var; attached to model
# invocations via with_config(callbacks=[...]), so events are recorded
# wherever the model is invoked (orchestrator, reviewers, subagent loops).

class DeepAgentTracer(BaseCallbackHandler):
    """Records LLM invocation, tool call, and chain events for debugging."""

    def __init__(self):
        super().__init__()
        self._events: list[dict] = []

    def _record(self, event_type: str, **data):
        data.setdefault("timestamp", datetime.now().isoformat())
        self._events.append({"type": event_type, **data})

    @staticmethod
    def _name(serialized: dict | None, kwargs) -> str:
        return (serialized or {}).get("name") or kwargs.get("name") or "unknown"

    # Chat models (ChatAnthropic/ChatOpenAI/etc. all emit chat_model events)
    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs):
        self._record(
            "chat_model_start",
            name=self._name(serialized, kwargs),
            message_count=len(messages or []),
        )

    def on_chat_model_end(self, response, **kwargs):
        self._record("chat_model_end")

    # Non-chat LLMs, for parity (on_llm_* fires for BaseLLM subclasses)
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self._record("llm_start", name=self._name(serialized, kwargs))

    def on_llm_end(self, response, **kwargs):
        self._record("llm_end")

    # Tool executions
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self._record("tool_start", name=self._name(None, kwargs))

    def on_tool_end(self, output, **kwargs):
        self._record("tool_end", name=self._name(None, kwargs),
                     output_length=len(str(output)))

    # Chain / graph steps
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        keyset = list(inputs.keys()) if isinstance(inputs, dict) else []
        self._record("chain_start", name=self._name(serialized, kwargs),
                     input_keys=keyset)

    def on_chain_end(self, outputs, **kwargs):
        self._record("chain_end")

    def get_events(self) -> list[dict]:
        return list(self._events)

    def clear(self):
        self._events.clear()


_tracer = DeepAgentTracer() if os.getenv("OBSERVABILITY") == "1" else None


def get_tracer() -> DeepAgentTracer | None:
    """Return the global tracer instance, or None if observability is disabled."""
    return _tracer


def _maybe_attach_callbacks(model):
    """Wrap a model with the observability callback when enabled (8.3).

    ``with_config`` merges the callbacks into every downstream invoke, so
    orchestrator, reviewer, and subagent-loop calls all record events without
    each call site passing callbacks itself.
    """
    if _tracer is not None:
        return model.with_config(callbacks=[_tracer])
    return model


# --- Model Selection ---
def get_model():
    """Return the configured model, with the observability callback attached
    when OBSERVABILITY=1 (8.3). Provider clients are cached per configuration
    (8.5); the callback wrap is cheap and applied per call."""
    return _maybe_attach_callbacks(_get_cached_model())


_model_cache: dict[tuple, object] = {}


def _cache_key(provider: str, model_name: str) -> tuple:
    return (provider, model_name)


def _get_cached_model():
    """Build one provider client per (provider, model) configuration.

    The cache lives for the process lifetime and is cleared by
    reset_deep_agent(), so a config change requires an explicit reset — same
    contract as the compiled-graph cache.

    OpenRouter-only for now. The full fallback chain (Anthropic → OpenRouter
    → OpenAI → Google → Ollama) is kept below, commented out, because the
    environment of a tool like Claude Code exports its own
    ``ANTHROPIC_API_KEY``/``ANTHROPIC_BASE_URL`` — which load_dotenv() cannot
    override — so the Anthropic-first order silently hijacked model selection
    and every call failed against that router. Restore the chain by swapping
    the two blocks back; the imports at the top of this file already cover
    the other providers.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")
    if not openrouter_key or not model_name:
        raise RuntimeError(
            "OPENROUTER_API_KEY and OPENROUTER_MODEL must both be set in .env. "
            "There is no fallback model: the model id must be configured "
            "explicitly. (The OpenRouter-only selection is active; other "
            "providers are commented out in agent_factory._get_cached_model.)"
        )
    key = _cache_key("openrouter", model_name)
    if key not in _model_cache:
        _model_cache[key] = ChatOpenRouter(
            model=key[1], api_key=openrouter_key, temperature=0,
        )
    return _model_cache[key]

    # --- Multi-provider fallback chain, disabled while OpenRouter-only ---
    # anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    # openai_key = os.getenv("OPENAI_API_KEY")
    # google_key = os.getenv("GOOGLE_API_KEY")
    #
    # if anthropic_key:
    #     key = _cache_key("anthropic", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"))
    #     if key not in _model_cache:
    #         _model_cache[key] = ChatAnthropic(model=key[1], temperature=0)
    #     return _model_cache[key]
    # elif openrouter_key:
    #     key = _cache_key("openrouter", os.getenv("OPENROUTER_MODEL", "stealth/space-bunny-alpha"))
    #     if key not in _model_cache:
    #         _model_cache[key] = ChatOpenRouter(
    #             model=key[1], api_key=openrouter_key, temperature=0,
    #         )
    #     return _model_cache[key]
    # elif openai_key:
    #     key = _cache_key("openai", os.getenv("OPENAI_MODEL", "gpt-4o"))
    #     if key not in _model_cache:
    #         _model_cache[key] = ChatOpenAI(model=key[1], temperature=0)
    #     return _model_cache[key]
    # elif google_key:
    #     key = _cache_key("google", os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
    #     if key not in _model_cache:
    #         _model_cache[key] = ChatGoogleGenerativeAI(model=key[1], temperature=0)
    #     return _model_cache[key]
    # else:
    #     key = _cache_key("ollama", "gemma4:12b-mlx")
    #     if key not in _model_cache:
    #         _model_cache[key] = ChatOllama(model="gemma4:12b-mlx", temperature=0.0)
    #     return _model_cache[key]


# --- Graph Construction ---

def get_deep_agent():
    """Return the compiled deep agent graph.

    The graph itself is owned by ``graph.get_deep_agent`` so there is exactly
    one compiled instance; this wrapper keeps the workspace/skills bootstrap
    that the ``task`` tool and the Streamlit app depend on.
    """
    from src.services.agent_orchestrator.graph import get_deep_agent as _compile

    workspace_root = str(get_workspace_root())
    if not os.path.exists(workspace_root):
        os.makedirs(workspace_root)

    skills_root = "./skills"
    if not os.path.exists(skills_root):
        os.makedirs(skills_root)

    return _compile()


def clear_caches():
    """Drop cached model clients and MCP tool registries.

    The compiled graph is owned by ``graph.py``; use
    ``graph.reset_deep_agent()`` (re-exported here) to clear everything.
    """
    _model_cache.clear()
    clear_mcp_client_cache()
    clear_mcp_bridge_cache()


def reset_deep_agent():
    """Invalidate the cached compiled graph, model clients, and MCP tools.

    Call this when tools, skills, system prompts, or model configuration
    change and you need a fresh graph. The next call to ``get_deep_agent()``
    will compile a new instance and rebuild provider clients (8.5), MCP
    tools (10.1), and the tool registry the sub-agent tool loop dispatches
    through — a stale registry would keep offering tools that no longer exist.
    """
    from src.services.agent_orchestrator.graph import reset_deep_agent as _reset
    from src.services.agent_orchestrator import subagents

    global _compiled_graph
    _compiled_graph = None
    subagents._TOOL_REGISTRY = None
    _reset()
