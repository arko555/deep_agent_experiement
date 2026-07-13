# Deep Agent Demo — Modular LangGraph Architecture

A production-grade **Deep Agent** built with LangChain and LangGraph. Features a modular, hierarchical multi-agent system with dynamic skill loading, specialized subagents, strict workspace isolation, and advanced governance controls.

---

## Architecture Overview

The system follows a hierarchical orchestration pattern where a central orchestrator manages specialized subagents and tools through a stateful graph.

```text
agent.py  (facade)
    └── src/core/agent_factory.py   ← Graph builder, tool definitions, and routing
            ├── src/state.py        ← AgentState (Messages, Plan, Audit Log, Metrics)
            ├── src/core/memory.py  ← System prompts, skills loader, AGENTS.md reader
            ├── src/core/rag.py     ← Tavily web search integration
            ├── src/core/guardrails.py  ← Path security and workspace isolation
            └── src/nodes/
                    ├── plan.py     ← Orchestrator node + `write_todos` tool
                    ├── research.py ← Researcher subagent (Structured Output)
                    ├── write.py    ← Writer subagent (Structured Output)
                    └── review.py   ← Critic (Adversarial) + Plan Checker nodes
```

### Core Principles

1.  **Generic Orchestrator** — A universal agent that decomposes tasks using the `write_todos` planning tool before acting.
2.  **Role-Specialized Subagents** — The `task` tool spawns isolated subagents with role-specific system prompts (`research`, `writer`) using **Structured Output** to ensure reliable data exchange.
3.  **Dynamic Skill Loading** — Specialized behaviors are defined as `SKILL.md` files in `skills/`. Agents discover available skills via metadata and load full instructions on-demand.
4.  **Strict Workspace Isolation** — All file outputs are constrained to `./workspace/` by path guardrails.
5.  **Governance & Control** —
    *   **Adversarial Review**: A `critic` node validates agent reasoning.
    *   **Plan Compliance**: A `plan_checker` ensures the agent follows its own `current_plan`.
    *   **Human-in-the-Loop (HITL)**: Destructive actions (file writes/edits) require user approval in the UI.
    *   **Auditability**: Every tool call and decision is recorded in a structured `audit_log`.

---

## Module Reference

| File | Role |
|------|------|
| `agent.py` | Thin facade — re-exports `get_deep_agent()` |
| `app.py` | Streamlit UI — renders chat, thinking process, audit logs, and HITL controls |
| `src/state.py` | `AgentState` — shared state including `audit_log`, `token_usage`, and `current_plan` |
| `src/core/agent_factory.py` | Core: LLM selection, tool definitions, `StateGraph` construction, and routing logic |
| `src/core/memory.py` | Loads `AGENTS.md`, scans `skills/` metadata, and manages system prompts |
| `src/core/rag.py` | `internet_search()` — wraps Tavily API |
| `src/core/guardrails.py` | `validate_and_normalize_path()` — enforces read/write boundaries |
| `src/nodes/plan.py` | `call_orchestrator()` node; `write_todos` tool |
| `src/nodes/research.py` | System prompt factory for the **Researcher** subagent (returns structured `ResearchResult`) |
| `src/nodes/write.py` | System prompt factory for the **Writer** subagent (returns structured `WriteResult`) |
| `src/nodes/review.py` | `call_agent_node`, `call_responder_node`, `call_critic_node`, and `call_plan_checker_node` |
| `skills/` | Directory containing `SKILL.md` files for dynamic capability loading |
| `tools/` | Directory for custom, dynamically loaded LangChain tools |
| `AGENTS.md` | Shared project conventions and entity definitions injected into prompts |

---

## Graph Construction

The LangGraph `StateGraph` implements a reasoning loop with validation:

```text
START
  │
  ▼
[orchestrator]  ──── LLM decides: tool calls? ────┐
  │                                                │
  │ No (final answer)              Yes (tool calls)│
  ▼                                                ▼
[critic] <────────────────────────────────────── [agent]
  │                                                │
  │ Approved?                                      ▼
  │                                             [tools]
  ▼                                                │
[responder] <──────────────────────────────────────┘
  │
  ▼
 END
```

### Routing Logic
1.  **Orchestrator** $\rightarrow$ **Agent**: If `tool_calls` are present.
2.  **Orchestrator** $\rightarrow$ **Critic**: If no tool calls, to validate the proposed response.
3.  **Critic** $\rightarrow$ **Orchestrator**: If the response is flagged for review or needs refinement.
4.  **Critic** $\rightarrow$ **Responder**: If the response is "APPROVED".

---

## Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic, OpenAI, or Ollama API Key
- Tavily API Key (for web search)

### Setup
```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application
**Streamlit UI (Recommended):**
```bash
uv run streamlit run app.py
```

**CLI Usage:**
```python
from agent import get_deep_agent
from langchain_core.messages import HumanMessage

agent = get_deep_agent()
result = agent.invoke({
    "messages": [HumanMessage(content="Research the latest trends in AI agents.")],
    "current_plan": [],
    "workspace_files": [],
    "audit_log": [],
    "token_usage": {},
    "iteration_count": 0,
    "max_iterations": 10,
    "max_tokens": 5000
})
print(result["messages"][-1].content)
```
