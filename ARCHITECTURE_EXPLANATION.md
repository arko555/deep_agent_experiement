# Deep Agent — Complete User Query Flow & Codebase Map

This document traces a user query from the moment it enters the Streamlit UI to the final response, mapping every function, every file, and every decision point. It also catalogs edge cases and scenarios the code does not currently handle.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point — `app.py`](#2-entry-point--apppy)
3. [Agent Factory — `agent_factory.py`](#3-agent-factory--agentfactorypy)
4. [Graph Nodes](#4-graph-nodes)
5. [Supporting Modules](#5-supporting-modules)
6. [State Schema — `state.py`](#6-state-schema--statepy)
7. [Complete Query Flow](#7-complete-query-flow)
8. [Edge Cases & Gaps](#8-edge-cases--gaps)

---

## 1. High-Level Architecture

```
User Input (Streamlit)
        │
        ▼
┌─────────────────────────────────────────┐
│  StateGraph (LangGraph)                 │
│                                         │
│  [orchestrator] ──tool calls?──► [agent]│
│       │                                │    │
│       │ No                           [tools]│
│       ▼                                │    │
│  [critic] ──approved?──► [plan_checker] │    │
│       │                                │    │
│       ▼                                │    │
│  [responder] ──────────────────────────┘    │
└─────────────────────────────────────────────┘
        │
        ▼
  Final message returned to Streamlit
```

**Files involved:** `app.py` → `agent.py` → `src/core/agent_factory.py` → `src/nodes/*.py` → `src/core/memory.py` → `src/core/guardrails.py` → `src/core/rag.py`

---

## 2. Entry Point — `app.py`

### 2.1 Initialization (module load)

| Line | Code | What it does |
|------|------|-------------|
| 11-18 | `logging.basicConfig(...)` | Sets up file + console logging |
| 22 | `load_dotenv()` | Loads `.env` into `os.environ` (API keys, model names, etc.) |
| 25-30 | `st.set_page_config(...)` | Configures Streamlit page metadata |
| 33-35 | `os.makedirs(workspace_dir)` | Ensures `./workspace/` exists |
| 38-104 | Custom CSS injection | Dark-theme styling for chat bubbles, sidebar, skill cards |

### 2.2 Session State (lines 107-125)

| Key | Default | Purpose |
|-----|---------|---------|
| `messages` | `[]` | Chat history list of `{"role": "user"|"assistant", "content": str}` |
| `agent` | `None` | Compiled LangGraph agent (created once, reused) |
| `current_plan` | `[]` | List of todo strings from `write_todos` |
| `workspace_files` | `[]` | Paths of files in `./workspace/` |
| `audit_log` | `[]` | Structured log entries: `{timestamp, action, details}` |
| `token_usage` | `{"total": 0}` | Cumulative token counts |
| `iteration_count` | `0` | Loop iterations in the graph |
| `last_action` | `"None"` | Human-readable description of last action |
| `current_node` | `"Idle"` | Current node being executed |

### 2.3 Sidebar Rendering (lines 166-268)

- **Governance** (lines 172-176): Displays `iteration_count` and `token_usage` metrics.
- **Active Skills** (lines 180-194): Scans `./skills/`, parses `SKILL.md` frontmatter via `get_skill_info()`, renders skill cards.
- **Audit Log** (lines 198-211): Reverse-ordered list of `st.session_state.audit_log` entries.
- **Shared Memory** (lines 220-225): Reads and displays `AGENTS.md`.
- **Workspace Files** (lines 229-259): Lists files, provides expandable previews and download buttons.

### 2.4 Chat Rendering & Query Processing (lines 270-405)

| Step | Lines | Function | What it does |
|------|-------|----------|-------------|
| 1 | 275-277 | Chat history loop | Iterates `st.session_state.messages`, renders each via `st.chat_message(role)` |
| 2 | 280 | `st.chat_input()` | Waits for user input; returns `prompt` string |
| 3 | 282 | Append user message | `st.session_state.messages.append({"role": "user", "content": prompt})` |
| 4 | 283-284 | Render user message | `with st.chat_message("user"): st.markdown(prompt)` |
| 5 | 292-294 | Thinking container | Creates `st.status("Agent is thinking...")` |
| 6 | 297-394 | Agent streaming | `st.session_state.agent.stream(..., stream_mode="updates")` — yields state updates per node |
| 7 | 307-338 | Message extraction | For each `data["messages"]`, iterates messages, maps roles, displays in thinking container |
| 8 | 340-370 | Tool call display | Extracts `tool_calls`, shows delegations, plan updates, skill loads, tool executions |
| 9 | 372-391 | State field updates | Updates `current_plan`, `audit_log`, `token_usage`, `iteration_count` from state diffs |
| 10 | 394-399 | Completion/error | `status.update(label="Process Complete!")` or error trace |
| 11 | 402-403 | Chat history append | Appends `turn_messages` (only responder messages) to `st.session_state.messages` |
| 12 | 405 | Workspace refresh | Calls `update_workspace_files()` |

### 2.5 Helper Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `update_workspace_files()` | 128-136 | Walks `./workspace/`, updates `st.session_state.workspace_files` |
| `get_skill_info(skill_path)` | 138-153 | Parses `SKILL.md` frontmatter (YAML between `---` delimiters) |
| `add_audit_entry(action, details)` | 155-163 | Appends timestamped entry to `st.session_state.audit_log` |

---

## 3. Agent Factory — `agent_factory.py`

### 3.1 Model Selection (`get_model()`, lines 33-64)

**Purpose:** Picks the LLM based on which API key is available.

**Priority order:**
1. Anthropic (`ANTHROPIC_API_KEY`) → `ChatAnthropic`
2. OpenRouter (`OPENROUTER_API_KEY`) → `ChatOpenRouter`
3. OpenAI (`OPENAI_API_KEY`) → `ChatOpenAI`
4. Google (`GOOGLE_API_KEY`) → `ChatGoogleGenerativeAI`
5. Ollama (fallback) → `ChatOllama`

**Edge case:** If multiple keys are set, only the highest-priority one is used. No error if no key is set — falls through to Ollama which may fail if no local server runs.

### 3.2 Tool Definitions (lines 66-168)

| Tool | Lines | Description |
|------|-------|-------------|
| `internet_search` | 68-81 | Wraps `src/core/rag.py:internet_search()` via Tavily API |
| `read_file` | 83-93 | Reads file after path validation via `guardrails.validate_and_normalize_path()` |
| `write_file` | 95-105 | Writes file to `./workspace/` after path normalization |
| `edit_file` | 107-127 | Searches-replaces in file within workspace |
| `task` | 129-158 | **Delegates to subagent** — spawns recursive agent or calls structured LLM |
| `list_tools` | 160-168 | Returns formatted list of all available tools |

**Critical detail — `task` tool (lines 129-158):**
- If `subagent_type` is `"research"` or `"researcher"`: calls LLM with `ResearchResult` structured output.
- If `subagent_type` is `"write"` or `"writer"`: calls LLM with `WriteResult` structured output.
- Otherwise: invokes a **recursive** `get_deep_agent()` instance with `HumanMessage(content=description)`.
- **Edge case:** The recursive call creates a new compiled graph each time (no memoization). This is expensive.

### 3.3 Dynamic Tool Loading (`load_dynamic_tools()`, lines 172-191)

Scans `./tools/` directory for Python modules, imports them, reloads, and extracts functions decorated with `@tool` (LangChain tool marker).

### 3.4 Graph Wrapper Nodes (lines 193-248)

| Wrapper | Lines | Delegates To |
|---------|-------|-------------|
| `local_orchestrator_node` | 195-197 | `call_orchestrator(state, model, tools)` |
| `local_agent_node` | 199-200 | `call_agent_node(state)` |
| `local_responder_node` | 202-203 | `call_responder_node(state)` |
| `local_tools_node` | 205-248 | Executes tool calls from last AIMessage |

### 3.5 Routing Logic (lines 250-284)

| Router | Lines | Logic |
|--------|-------|-------|
| `route_from_orchestrator` | 251-264 | If `next_message` has `tool_calls` → `"agent"`, else → `"critic"` |
| `route_from_critic` | 266-274 | Always returns `"orchestrator"` (hardcoded; "APPROVED" check is dead code) |
| `route_from_plan_checker` | 276-284 | Always returns `"orchestrator"` (same issue) |

### 3.6 Graph Construction (`get_deep_agent()`, lines 301-336)

**Purpose:** Builds and compiles the `StateGraph`.

**Nodes added:**
- `orchestrator` → `local_orchestrator_node`
- `agent` → `local_agent_node`
- `responder` → `local_responder_node`
- `tools` → `local_tools_node`
- `critic` → `call_critic_node(state, model)`
- `plan_checker` → `call_plan_checker_node(state, model)`

**Edges:**
- `START → orchestrator`
- `orchestrator → {agent, critic, responder}` (conditional)
- `agent → tools`
- `tools → orchestrator`
- `critic → orchestrator`
- `responder → END`

**Critical observation:** There is **no conditional edge** from `critic` or `plan_checker` to `responder`. The critic always routes back to `orchestrator`, which means the `responder` node is only reachable if the orchestrator itself directly returns a message with no tool calls (which shouldn't happen given the routing logic). This is a **structural bug** — the graph can never reach the responder node through the critic path.

---

## 4. Graph Nodes

### 4.1 Orchestrator — `src/nodes/plan.py`

| Function | Lines | What it does |
|----------|-------|-------------|
| `write_todos` (tool) | 7-14 | Returns "Updated todo list" string (doesn't actually update state — state is handled by `local_tools_node`) |
| `call_orchestrator` | 16-32 | Prepends system prompt + all messages, calls LLM with tools bound, returns `{"next_message": response}` |

**Key detail:** The orchestrator returns `next_message`, not `messages`. This means the LLM response is stored for the next node's consumption, not appended to the conversation yet.

### 4.2 Agent Node — `src/nodes/review.py`

| Function | Lines | What it does |
|----------|-------|-------------|
| `call_agent_node` | 15-19 | Returns `{"messages": [state["next_message"]], "next_message": None}` |

**Purpose:** Extracts the orchestrator's response from `next_message` and puts it in `messages` for the tools node to execute.

### 4.3 Responder Node — `src/nodes/review.py`

| Function | Lines | What it does |
|----------|-------|-------------|
| `call_responder_node` | 21-25 | Returns `{"messages": [state["next_message"]], "next_message": None}` |

**Purpose:** Extracts the critic's "approved" response and puts it in `messages` for the graph to terminate.

### 4.4 Tools Node — `agent_factory.py:local_tools_node()`

| Lines | What it does |
|-------|-------------|
| 206-207 | Gets last message, extracts `tool_calls` |
| 213-218 | Initializes audit entry |
| 220-242 | For each tool call: validates tool exists, invokes it, creates `ToolMessage` |
| 244-248 | Returns: `{"messages": tool_messages, "workspace_files": ..., "audit_log": [...], "iteration_count": incremented}` |

### 4.5 Critic Node — `src/nodes/review.py`

| Lines | What it does |
|-------|-------------|
| 31 | Gets last message from state |
| 37-42 | System prompt: "Find flaws in the agent's response" |
| 45-48 | Calls LLM with system prompt + state messages |
| 53-56 | If "APPROVED" → returns `{"next_message": response, "messages": [response]}` |

### 4.6 Plan Checker Node — `src/nodes/review.py`

| Lines | What it does |
|-------|-------------|
| 62-64 | If no plan → returns `{"next_message": None}` |
| 66-76 | System prompt: "Check if response aligns with plan" |
| 79-80 | If "COMPLIANT" → returns `{"next_message": None}` |

---

## 5. Supporting Modules

### 5.1 State — `src/state.py`

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]   # append-only
    current_plan: List[str]                        # from write_todos
    workspace_files: List[str]                     # synced after tool exec
    next_message: Optional[BaseMessage]            # staging area
    subagent_role: Optional[str]                   # "research"/"writer"
    audit_log: Annotated[List[Dict], add]          # action log
    token_usage: Dict[str, int]                    # API consumption
    iteration_count: int                           # loop count
    max_iterations: int                            # safety limit
    max_tokens: int                                # budget limit
```

The `add` annotation means LangGraph will **append** to these fields rather than replace them.

### 5.2 Memory — `src/core/memory.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `MemoryManager` class | 6-42 | Thread/entity file-based memory (not actually used by the main flow) |
| `get_workspace_files()` | 47-56 | Walks `./workspace/`, returns sorted file list |
| `get_skill_info()` | 58-76 | Parses SKILL.md frontmatter |
| `get_skills_summary()` | 78-90 | Returns bullet list of available skills |
| `get_tools_summary()` | 92-103 | Returns bullet list of tool descriptions |
| `get_memory_content()` | 105-113 | Reads `AGENTS.md` |
| `get_system_prompt()` | 115-153 | Builds orchestrator system prompt with skills + tools + AGENTS.md |

### 5.3 Guardrails — `src/core/guardrails.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `validate_and_normalize_path()` | 3-32 | Prevents path traversal; restricts writes to `./workspace/` |

### 5.4 RAG — `src/core/rag.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `internet_search()` | 4-28 | Wraps Tavily API for web search |

---

## 6. Complete Query Flow

### Scenario: User asks "Research multi-agent systems and write a blog post"

#### Step 1: User submits prompt
- **File:** `app.py`
- **Function:** `st.chat_input()` (line 280)
- **Action:** `prompt = st.chat_input(...)` captures user text

#### Step 2: User message added to history
- **File:** `app.py`
- **Lines:** 282-284
- **Action:** `st.session_state.messages.append({"role": "user", "content": prompt})`

#### Step 3: Agent streaming begins
- **File:** `app.py`
- **Lines:** 297-394
- **Action:** `st.session_state.agent.stream({"messages": st.session_state.messages}, stream_mode="updates")`

#### Step 4: Orchestrator node executes
- **File:** `src/nodes/plan.py`
- **Function:** `call_orchestrator(state, model, tools)`
- **Lines:** 16-32
- **Action:**
  1. Calls `get_system_prompt()` → builds prompt with skills summary, tools summary, AGENTS.md
  2. Prepends system message to all `state["messages"]`
  3. Binds tools to model: `model.bind_tools(tools)`
  4. Calls `model_with_tools.invoke(formatted_messages)`
  5. Returns `{"next_message": response}` where response is an AIMessage with tool calls

#### Step 5: Agent node extracts orchestrator response
- **File:** `src/nodes/review.py`
- **Function:** `call_agent_node(state)`
- **Lines:** 15-19
- **Action:** Returns `{"messages": [state["next_message"]], "next_message": None}`

#### Step 6: Routing — orchestrator → agent (tool calls present)
- **File:** `src/core/agent_factory.py`
- **Function:** `route_from_orchestrator(state)`
- **Lines:** 251-264
- **Action:** Checks `next_msg.tool_calls`, returns `"agent"`

#### Step 7: Tools node executes tool calls
- **File:** `src/core/agent_factory.py`
- **Function:** `local_tools_node(state)`
- **Lines:** 205-248
- **Action:**
  1. Iterates `last_message.tool_calls`
  2. For each tool: invokes it, creates `ToolMessage`
  3. Returns `{"messages": tool_messages, "workspace_files": ..., "audit_log": [...], "iteration_count": n+1}`

#### Step 8: Orchestrator routes again
- **File:** `src/core/agent_factory.py`
- **Function:** `route_from_orchestrator(state)`
- **Action:** Now `next_message` is None (cleared by agent node), so routes to `"critic"`

#### Step 9: Critic evaluates response
- **File:** `src/nodes/review.py`
- **Function:** `call_critic_node(state, model)`
- **Lines:** 27-56
- **Action:**
  1. Gets last message from state
  2. Calls LLM with critique prompt
  3. Returns `{"next_message": response, "messages": [response]}`

#### Step 10: Critic routes back to orchestrator
- **File:** `src/core/agent_factory.py`
- **Function:** `route_from_critic(state)`
- **Lines:** 266-274
- **Action:** Always returns `"orchestrator"` (regardless of APPROVED status)

#### Step 11: Orchestrator evaluates again
- **File:** `src/nodes/plan.py`
- **Function:** `call_orchestrator(state, model, tools)`
- **Action:** LLM sees critic's feedback + full history, generates new response with/without tool calls

#### Step 12: Loop continues until orchestrator returns no tool calls
- The orchestrator keeps looping through agent → tools → orchestrator until it produces an AIMessage with no `tool_calls`.

#### Step 13: Final response captured in Streamlit
- **File:** `app.py`
- **Lines:** 328-338
- **Action:** When `node_name == "responder"`, the message is captured in `turn_messages` and appended to `st.session_state.messages`

#### Step 14: Workspace files refreshed
- **File:** `app.py`
- **Lines:** 405
- **Action:** `update_workspace_files()` scans `./workspace/` for new files

---

## 7. Edge Cases & Gaps

### 7.1 Greetings ("hello", "hi", "good morning")

**What happens:**
1. User types "hello"
2. Agent is invoked with `messages = [{"role": "user", "content": "Research..."}, {"role": "assistant", "content": "<final response>"}, {"role": "user", "content": "hello"}]`
3. Orchestrator receives this conversation and, being a generic agent with no explicit "greeting" instruction, may:
   - Respond with a brief greeting (best case)
   - Try to execute tool calls from the first query again (if they weren't fully completed)
   - Ignore the greeting and provide a generic response

**Gap:** The orchestrator system prompt (line 126-147 in `memory.py`) has no instruction for handling simple greetings or chit-chat. It's designed as a "task executor" not a conversational agent.

### 7.2 Empty or minimal prompts

**What happens:**
- User types "" or "..." or a single word like "ok"
- Orchestrator receives ambiguous input, may produce unhelpful tool calls or generic responses

**Gap:** No input validation or length check in `app.py`. No guidance in system prompt about handling ambiguous requests.

### 7.3 Very long conversations

**What happens:**
- `state["messages"]` grows with each turn
- `call_orchestrator` prepends the system prompt to ALL messages
- Token count grows linearly, eventually exceeding model context limits

**Gap:** `state.py` defines `max_tokens` and `max_iterations` but `call_orchestrator` never checks them. No truncation or summarization logic exists.

### 7.4 Concurrent requests

**What happens:**
- Two users (or same user with two tabs) submit queries simultaneously
- Both use the same compiled `StateGraph` instance
- Streamlit's session state is per-session, so each tab has its own `messages` list

**Gap:** The compiled agent graph is created once per Streamlit session and shared across all requests. LangGraph's `StateGraph` is not thread-safe for concurrent invocations.

### 7.5 API failures

**What happens:**
- API key is invalid, rate-limited, or service is down
- `internet_search()` returns error string (graceful)
- LLM calls throw HTTP errors

**Gap:** `app.py` catches exceptions at line 395 but only shows a traceback. The agent is not restarted, so subsequent requests may also fail. No retry logic exists.

### 7.6 Recursive subagent explosion

**What happens:**
- User asks a complex multi-step task
- Orchestrator delegates to `task` tool with `general-purpose` subagent
- That subagent also calls `get_deep_agent()` and creates its own graph
- Each recursion level creates a new compiled graph

**Gap:** No depth limit on recursion. `max_iterations` is per-graph, not per-recursion. A deeply nested chain of subagents could exhaust memory.

### 7.7 `edit_file` tool bug

**File:** `agent_factory.py`, lines 107-127

```python
new_content = content.replace(search_text, replace_text) # line 121
new_content = content.replace(search_text, replace_text) # line 122 (duplicate, overwrites)
```

**What happens:** The `edit_file` tool has redundant replacement logic. Lines 121-122 call `replace()` twice on `content` (not `new_content`), so the first replacement's result is lost. The file ends up with only the second replacement applied.

### 7.8 Critic routing dead code

**File:** `agent_factory.py`, lines 266-274

```python
def route_from_critic(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "responder"
    if "APPROVED" in next_msg.content.upper():
        return "orchestrator"
    else:
        return "orchestrator"
```

**What happens:** Both branches return `"orchestrator"`. The `if "APPROVED"` check is dead code — it never routes to `"responder"` even when the critic approves the response. The graph can never reach the `responder` node through the critic path.

### 7.9 Plan checker dead code

**File:** `agent_factory.py`, lines 276-284

```python
def route_from_plan_checker(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "orchestrator"
    if "COMPLIANT" in next_msg.content.upper():
        return "orchestrator"
    else:
        return "orchestrator"
```

**What happens:** Same issue — always returns `"orchestrator"`. The `plan_checker` node is never actually reachable because no edge routes to it. It's defined in the graph (line 317) but has no incoming conditional edge from the orchestrator.

### 7.10 `subagent_role` state field unused

**File:** `state.py`, field `subagent_role: Optional[str]`

**What happens:** Defined in `AgentState` but never read or written by any node. The `task` tool uses `subagent_type` from tool args, not `state["subagent_role"]`.

### 7.11 MemoryManager class unused

**File:** `src/core/memory.py`, class `MemoryManager` (lines 6-42)

**What happens:** The class provides thread/entity persistence but is never instantiated or called anywhere in the codebase. All memory is handled through `st.session_state` in Streamlit.

### 7.12 No system prompt for responder/agent nodes

**What happens:** `call_agent_node` and `call_responder_node` don't set any system prompt. They just pass through `next_message`. If the last message in `next_message` is the critic's evaluation (not the original response), the responder would forward the critic's opinion as the final answer.

### 7.13 Streamlit re-render duplicates messages

**What happens:** When `st.rerun()` is called (e.g., after HITL approval at line 369), the entire page re-renders. The `for message in st.session_state.messages` loop (line 275) renders all messages. If `st.session_state.messages` was modified during the same request (which Streamlit does), the new messages appear twice — once from the live rendering and once from the history loop.

**Current mitigation:** Only responder messages are captured in `turn_messages`, reducing duplication risk. But the initial user message added at line 282 will be rendered by both the `st.chat_message("user")` block (line 283) and the history loop (line 275) on the next render.

---

## 8. Function Call Map

### `app.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `update_workspace_files()` | Sidebar (line 231), query handler (line 405) | Refresh workspace file list |
| `get_skill_info()` | Sidebar (line 187) | Parse SKILL.md frontmatter |
| `add_audit_entry()` | Tool call handler (line 360, 368) | Log actions to audit log |

### `agent.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `get_deep_agent()` | `app.py` (line 111), `task` tool (line 139) | Create/return compiled LangGraph agent |

### `src/core/agent_factory.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `get_model()` | `get_deep_agent()` (line 312), `call_critic_node()` (line 316), `call_plan_checker_node()` (line 317) | Select LLM backend |
| `load_dynamic_tools()` | `get_all_tools()` (line 298) | Scan `./tools/` for @tool functions |
| `get_all_tools()` | `local_orchestrator_node()` (line 196), `local_tools_node()` (line 211) | Combine built-in + dynamic tools |
| `local_orchestrator_node()` | Graph (line 312) | Wrap `call_orchestrator` for graph |
| `local_agent_node()` | Graph (line 313) | Wrap `call_agent_node` for graph |
| `local_responder_node()` | Graph (line 314) | Wrap `call_responder_node` for graph |
| `local_tools_node()` | Graph (line 315) | Execute tool calls from AIMessage |
| `route_from_orchestrator()` | Graph conditional edge (line 321) | Decide agent/critic/responder |
| `route_from_critic()` | Graph conditional edge (line 321) | Always returns orchestrator |
| `route_from_plan_checker()` | Graph conditional edge (line 321) | Always returns orchestrator |
| `get_deep_agent()` | Graph construction (line 301) | Build and compile StateGraph |

### `src/nodes/plan.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `write_todos` (tool) | Graph (built-in tools) | Return "Updated todo list" |
| `call_orchestrator()` | `local_orchestrator_node()` (line 197) | Call LLM with tools, return next_message |

### `src/nodes/review.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `get_text_content()` | `call_critic_node()` (line 49), `call_plan_checker_node()` (line 77) | Extract text from multimodal content |
| `call_agent_node()` | `local_agent_node()` (line 200) | Extract next_message into messages |
| `call_responder_node()` | `local_responder_node()` (line 203) | Extract next_message into messages |
| `call_critic_node()` | Graph (line 316) | Evaluate response, return approved/rejected |
| `call_plan_checker_node()` | Graph (line 317) | Check plan compliance |

### `src/core/memory.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `get_workspace_files()` | `local_tools_node()` (line 245), `app.py` (line 129) | List workspace files |
| `get_skill_info()` | `get_skills_summary()` (line 85) | Parse SKILL.md |
| `get_skills_summary()` | `get_system_prompt()` (line 123) | Build skills list |
| `get_tools_summary()` | `get_system_prompt()` (line 124) | Build tools list |
| `get_memory_content()` | `get_system_prompt()` (line 149) | Read AGENTS.md |
| `get_system_prompt()` | `call_orchestrator()` (line 28) | Build orchestrator system prompt |

### `src/core/guardrails.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `validate_and_normalize_path()` | `read_file` (line 87), `write_file` (line 99), `edit_file` (line 111) | Prevent path traversal |

### `src/core/rag.py`
| Function | Called From | Purpose |
|----------|-------------|---------|
| `internet_search()` | `internet_search` tool (line 81) | Call Tavily API |

---

## 9. Summary of Issues Found

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | **Critical** | `agent_factory.py:266-274` | `route_from_critic` always returns `"orchestrator"` — dead code |
| 2 | **Critical** | `agent_factory.py:276-284` | `route_from_plan_checker` always returns `"orchestrator"` — dead code |
| 3 | **High** | `agent_factory.py:121-122` | `edit_file` has duplicate replace, first result overwritten |
| 4 | **Medium** | `memory.py:6-42` | `MemoryManager` class defined but never used |
| 5 | **Medium** | `state.py:10` | `subagent_role` field defined but never used |
| 6 | **Medium** | `app.py:282` | User message rendered twice (live + history loop) on re-render |
| 7 | **Medium** | No context truncation | Long conversations exceed model context limits |
| 8 | **Low** | No greeting handling | No guidance for chit-chat in system prompt |
| 9 | **Low** | No input validation | Empty/minimal prompts accepted without check |
| 10 | **Low** | No retry logic | API failures not retried |
| 11 | **Low** | No recursion depth limit | Subagent delegation can recurse infinitely |
