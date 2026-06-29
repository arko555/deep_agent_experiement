# Deep Agent Demo — Modular LangGraph Architecture

A production-grade **Deep Agent** built with LangChain and LangGraph. Features a modular, hierarchical multi-agent system with dynamic skill loading, specialized subagents, and strict workspace isolation.

---

## Architecture Overview

```
agent.py  (facade)
    └── src/core/agent_factory.py   ← graph builder, tools, LLM selection
            ├── src/state.py        ← AgentState TypedDict
            ├── src/core/memory.py  ← system prompts, skills loader, AGENTS.md reader
            ├── src/core/rag.py     ← Tavily web search
            ├── src/core/guardrails.py  ← path security
            └── src/nodes/
                    ├── plan.py     ← orchestrator node + write_todos tool
                    ├── research.py ← Researcher subagent system prompt
                    ├── write.py    ← Writer subagent system prompt
                    └── review.py   ← agent (thought) + responder (final answer) nodes
```

### Core Principles

1. **Generic Orchestrator** — A universal agent that decomposes tasks using the `write_todos` planning tool before acting.
2. **Role-Specialized Subagents** — The `task` tool spawns isolated subagents with role-specific system prompts (`research`, `writer`) to keep the main context clean.
3. **Dynamic Skill Loading** — Specialized behaviors are defined as `SKILL.md` files in `skills/`. Agents discover available skills via metadata in the system prompt and load full instructions on-demand via `read_file`.
4. **Strict Workspace Isolation** — All file outputs are constrained to `./workspace/` by path guardrails. No writes to `/tmp/` or outside the repository.

---

## Module Reference

| File | Role |
|------|------|
| `agent.py` | Thin facade — re-exports `get_deep_agent()` for backward compatibility with `app.py` |
| `app.py` | Streamlit UI — streams graph events, renders thoughts, tool calls, plan, and workspace files |
| `src/state.py` | `AgentState` TypedDict — shared state flowing through every graph node |
| `src/core/agent_factory.py` | Core: LLM selection, tool definitions (guardrailed), `StateGraph` construction, routing logic |
| `src/core/memory.py` | Loads `AGENTS.md`, scans `skills/` metadata, builds role-aware system prompts |
| `src/core/rag.py` | `internet_search()` — wraps Tavily API with error handling |
| `src/core/guardrails.py` | `validate_and_normalize_path()` — enforces read/write boundaries |
| `src/nodes/plan.py` | `call_orchestrator()` node — invokes LLM with bound tools; `write_todos` tool definition |
| `src/nodes/research.py` | System prompt factory for the **Researcher** subagent role |
| `src/nodes/write.py` | System prompt factory for the **Writer** subagent role |
| `src/nodes/review.py` | `call_agent_node()` (surfaces thoughts to UI) and `call_responder_node()` (final answer) |
| `skills/research/SKILL.md` | Research skill spec — loaded dynamically by the agent on demand |
| `skills/writer/SKILL.md` | Writer skill spec — loaded dynamically by the agent on demand |
| `AGENTS.md` | Shared project conventions and entity definitions, injected into every system prompt |

---

## Graph Construction

The LangGraph `StateGraph` is built in `src/core/agent_factory.py → get_deep_agent()`:

```
START
  │
  ▼
[orchestrator]  ──── LLM decides: tool calls? ────┐
  │                                                │
  │ No (final answer)              Yes (tool calls)│
  ▼                                                ▼
[responder]                                     [agent]
  │                                                │
  ▼                                                ▼
 END                                           [tools]
                                                   │
                                                   └──────► [orchestrator]  (loop)
```

### Nodes

| Node | Function | Description |
|------|----------|-------------|
| `orchestrator` | `call_orchestrator()` | Prepends system prompt, binds tools to LLM, invokes model, stages response in `next_message` |
| `agent` | `call_agent_node()` | Commits `next_message` → `messages`; UI displays this as "Agent Thought" |
| `tools` | `local_tools_node()` | Executes all `tool_calls` from the last message, returns `ToolMessage` results |
| `responder` | `call_responder_node()` | Commits `next_message` → `messages`; UI captures this as the final answer |

### Routing

After `orchestrator` runs, the router inspects `next_message`:
- **Has `tool_calls`** → route to `agent` → `tools` → back to `orchestrator` (reasoning loop)
- **No `tool_calls`** → route to `responder` → `END`

---

## Query Flow — End to End

```
User prompt
    │
    ▼
app.py  →  agent.stream({"messages": [HumanMessage(...)]})
    │
    ▼
[orchestrator]
  · Builds system prompt (loads skills metadata + AGENTS.md)
  · Calls LLM with all tools bound
  · LLM may call write_todos → plan sidebar updates
  · LLM may call read_file("skills/research/SKILL.md") → loads skill instructions
    │
    ▼
[agent]  →  "Agent Thought:" displayed in Streamlit
    │
    ▼
[tools]  →  executes tool calls:
  · internet_search  → Tavily API
  · read_file        → loads skill/workspace files (path-guarded)
  · write_file       → saves to ./workspace/ (path-guarded)
  · edit_file        → modifies ./workspace/ files (path-guarded)
  · task             → spawns a subagent:
        └─ get_deep_agent() invoked recursively
           subagent_role="research" → Researcher system prompt
           subagent_role="writer"   → Writer system prompt
           runs its own full graph loop
           returns last message content as string
    │
    ▼
[orchestrator]  (loop continues until no more tool calls)
    │
    ▼
[responder]  →  final answer rendered in Streamlit chat
```

---

## AgentState

All graph nodes read from and write to `AgentState`:

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]  # append-only message log
    current_plan: List[str]                       # todo list (updated by write_todos)
    workspace_files: List[str]                    # synced after every tool execution
    next_message: Optional[BaseMessage]           # staging area for LLM response
    subagent_role: Optional[str]                  # "research", "writer", or None
```

---

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- One of: Anthropic API Key, OpenAI API Key, or a local [Ollama](https://ollama.com) instance
- Tavily API Key (for web search)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/arko555/deep_agent_experiement.git
cd deep_agent_experiement

# 2. Create .env
cp .env.example .env   # or create manually:
cat > .env << EOF
ANTHROPIC_API_KEY=your_anthropic_key   # preferred
# OPENAI_API_KEY=your_openai_key       # alternative
TAVILY_API_KEY=your_tavily_key
WORKSPACE_ROOT=./workspace
EOF

# 3. Install dependencies (uv handles the venv automatically)
uv sync
```

> **Model priority:** The agent picks the first available key: Anthropic → OpenAI → Ollama (local fallback, no key required).

### Run the Streamlit App

```bash
uv run streamlit run app.py
```

### Run from the CLI (no UI)

```python
from agent import get_deep_agent
from langchain_core.messages import HumanMessage

agent = get_deep_agent()
result = agent.invoke({
    "messages": [HumanMessage(content="Research the latest trends in AI agents and write a summary.")],
    "current_plan": [],
    "workspace_files": []
})
print(result["messages"][-1].content)
```

---

## Adding Skills

Create a new directory under `skills/` following the [Agent Skills spec](https://agentskills.io/specification):

```
skills/
└── my_skill/
    └── SKILL.md      ← YAML frontmatter + instructions
```

**`SKILL.md` template:**
```markdown
---
name: my_skill
description: >
  One-sentence description the agent uses to decide when to load this skill.
---

# My Skill

## Instructions
1. Step one...
2. Step two...
```

The agent automatically discovers new skills at runtime — no code changes needed.

---

## How to Test

Submit a complex prompt that requires multiple skills:
> *"Research the impact of multi-agent systems on software engineering and draft a 500-word blog post."*

Watch in the Streamlit sidebar:
- **Current Plan** — updates as `write_todos` is called
- **Active Skills** — lists discovered skills from `skills/`
- **Workspace Files** — shows files written to `./workspace/`
- **Shared Memory** — contents of `AGENTS.md`

---

## Project Conventions

See [`AGENTS.md`](./AGENTS.md) for shared project context, entity roles, and conventions that are automatically injected into every agent's system prompt.
