# Deep Agent Demo — Modular LangGraph Architecture

A production-grade **Deep Agent** framework built with LangChain and LangGraph. This project demonstrates a modular, hierarchical multi-agent system designed for high-reliability tasks requiring complex reasoning, specialized expertise, and strict governance.

---

## 🌟 Why This Architecture?

Standard LLM implementations often suffer from "context drift," where the model loses track of the original goal, or "hallucination," where it generates incorrect information without verification. The **Deep Agent** architecture solves these problems through several key design patterns:

### 🚀 Key Benefits

*   **High Accuracy & Reliability**: 
    *   **Adversarial Review**: A dedicated "Critic" node challenges the agent's reasoning before a final answer is delivered.
    *   **Plan Adherence**: A "Plan Checker" ensures the agent is actually completing the tasks it set for itself.
    *   **Structured Communication**: Subagents communicate via strict Pydantic schemas, preventing the "unstructured noise" that often breaks complex agentic loops.
*   **Enterprise-Grade Governance**:
    *   **Human-in-the-Loop (HITL)**: Critical actions like writing or editing files are paused for user approval, preventing unintended side effects.
    **Auditability**: Every single tool call, subagent delegation, and reasoning step is recorded in a structured `audit_log` for full transparency.
    *   **Resource Management**: Built-in tracking for token usage and iteration counts prevents runaway costs and infinite loops.
*   **Infinite Extensibility**:
    *   **Dynamic Skill Loading**: Add new capabilities simply by dropping a `SKILL.md` file into the `skills/` directory—no code changes required.
    *   **Modular Nodes**: New specialized roles (e.g., "Coder", "Legal Reviewer") can be added as new nodes in the LangGraph without refactoring the core orchestrator.

---

## 🛠️ Adaptability: Real-World Scenarios

This architecture is designed to be adapted to various complex workflows:

*   **Automated Research & Reporting**: Use the `research` and `writer` subagents to perform deep web searches and synthesize them into professional markdown reports.
*   **Software Engineering Assistant**: Extend the `tools/` directory with code execution and file manipulation tools to create an agent that can plan, write, and test code within a secure workspace.
*   **Content Creation Pipeline**: Orchestrate a workflow where one agent researches a topic, another outlines it, a third writes the draft, and a fourth performs a final editorial review.
*   **Data Analysis & Synthesis**: Integrate data retrieval tools and a "Data Analyst" subagent to transform raw data into structured insights and visualizations.

---

## 🏗️ Architecture Deep Dive

The system operates as a **Stateful Orchestration Loop**. Instead of a single long prompt, the task is decomposed into a series of discrete, verifiable steps.

### 1. The Orchestration Loop
The central `orchestrator` acts as the "brain." It doesn't do the heavy lifting; instead, it:
1.  **Plans**: Uses the `write_todos` tool to create a roadmap.
2.  **Delegates**: Uses the `task` tool to spawn specialized subagents.
3.  **Verifies**: Passes the results through a **Critic** and **Plan Checker** to ensure the work meets the required standards.

### 2. The Role of State (`AgentState`)
The `AgentState` is the "shared memory" and "control plane" of the entire system. It flows through every node in the graph and contains:
*   **`messages`**: The full conversation history.
*   **`current_plan`**: The dynamic list of tasks the agent is working through.
*   **`audit_log`**: A structured record of every action taken.
*   **`workspace_files`**: A real-time view of the files created or modified in the `./workspace/` directory.

### 3. Dynamic Capabilities
*   **Skills (`skills/`)**: These are "on-demand" instructions. The agent only loads a skill's full instructions when it realizes it needs that specific expertise, keeping the main context window clean and focused.
*   **Tools (`tools/`)**: These are the agent's "hands." They allow the agent to interact with the real world (web search, file system, etc.) within a strictly guarded environment.

---

## 🗺️ Module Reference

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

## 🔄 Graph Construction

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

## 🚀 Getting Started

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

---

## ➕ Adding Skills

Create a new directory under `skills/` following the [Agent Skills spec](https://agentskills.io/specification):

```text
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

## 🧪 How to Test

Submit a complex prompt that requires multiple skills:
> *"Research the impact of multi-agent systems on software engineering and draft a 500-word blog post."*

Watch in the Streamlit sidebar:
- **Current Plan** — updates as `write_todos` is called
- **Active Skills** — lists discovered skills from `skills/`
- **Workspace Files** — shows files written to `./workspace/`
- **Shared Memory** — contents of `AGENTS.md`

---

## 📋 Project Conventions

See [`AGENTS.md`](./AGENTS.md) for shared project context, entity roles, and conventions that are automatically injected into every agent's system prompt.
