"""Fake chat model for graph-level delegation tests (Phase 9).

``ScriptedChatModel`` is the same instance shared by every actor in a test
(the orchestrator, reviewers, and subagent loops all resolve their model via
``agent_factory.get_model()``). Because each actor's *system prompt* is
distinct, the model discriminates on ``messages[0]`` and hands back a canned
per-role script. This keeps a full compiled-graph run deterministic without a
real LLM.
"""

from collections import deque

from langchain_core.messages import AIMessage


def _usage(inp: int = 0, out: int = 0):
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


class ScriptedChatModel:
    """A ``.invoke``-able fake that pops canned responses per detected role.

    Roles are matched by a substring of the first message's content:
      - "generic Deep Agent"   → orchestrator
      - "Critical Reviewer"    → critic
      - "Plan Compliance"      → plan_checker
      - "Reflection Agent"     → reflection
      - "specialized subagent" → tool-loop subagent (research/writer)

    Each role maps to a list of canned AIMessages; the next one is returned on
    each call for that role (the last one repeats once exhausted). Every message
    carries ``usage_metadata`` so token aggregation can be asserted exactly.
    """

    def __init__(self, scripts: dict, default: AIMessage):
        self._scripts = {role: deque(msgs) for role, msgs in scripts.items()}
        self._default = default
        self._counts = {}  # role -> number of invokes

    # -- Runnable surface -------------------------------------------------
    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        role = self._detect_role(messages)
        self._counts[role] = self._counts.get(role, 0) + 1
        dq = self._scripts.get(role)
        if not dq:
            return self._default
        return dq.popleft() if len(dq) > 1 else dq[0]

    @property
    def counts(self):
        return dict(self._counts)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _detect_role(messages):
        first = messages[0] if messages else None
        content = getattr(first, "content", "")
        content = content if isinstance(content, str) else str(content)
        if "generic Deep Agent" in content:
            return "orchestrator"
        if "Critical Reviewer" in content:
            return "critic"
        if "Plan Compliance" in content:
            return "plan_checker"
        if "Reflection Agent" in content:
            return "reflection"
        if "specialized subagent" in content:
            return "subagent"
        return "other"


def ai(content, tool_calls=None, usage=(0, 0)):
    """Build a canned AIMessage with optional tool call(s) + usage metadata."""
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=_usage(*usage),
    )
