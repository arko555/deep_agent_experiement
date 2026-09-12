"""Deep agent graph factory.

Constructs and caches the LangGraph state graph for the deep agent workflow.
Routing, tools, and retry logic are imported from their dedicated modules.
"""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# pyrefly: ignore [missing-import]
from src.state import AgentState
# pyrefly: ignore [missing-import]
from src.core.config import (
    get_max_parallel_tasks,
    get_max_subagent_depth,
    get_subagent_timeout_seconds,
)
from src.core.subagents import is_parallelizable
from src.core.mcp_client import clear_mcp_tools_cache
from src.core.memory import get_workspace_files, get_system_prompt
from src.core.routing import (
    route_from_orchestrator,
    route_from_critic,
    route_from_plan_checker,
    route_from_reflection,
)
from src.core.utils import invoke_with_retry
from src.core.tools import get_all_tools, _execute_task
from src.nodes.plan import call_orchestrator
from src.nodes.review import (
    call_agent_node,
    call_responder_node,
    call_critic_node,
    call_plan_checker_node,
    call_reflection_node,
)

load_dotenv()


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
    contract as the compiled-graph cache."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if anthropic_key:
        key = _cache_key("anthropic", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"))
        if key not in _model_cache:
            _model_cache[key] = ChatAnthropic(model=key[1], temperature=0)
        return _model_cache[key]
    elif openrouter_key:
        key = _cache_key("openrouter", os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
        if key not in _model_cache:
            _model_cache[key] = ChatOpenRouter(model=key[1], api_key=openrouter_key, temperature=0)
        return _model_cache[key]
    elif openai_key:
        key = _cache_key("openai", os.getenv("OPENAI_MODEL", "gpt-4o"))
        if key not in _model_cache:
            _model_cache[key] = ChatOpenAI(model=key[1], temperature=0)
        return _model_cache[key]
    elif google_key:
        key = _cache_key("google", os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
        if key not in _model_cache:
            _model_cache[key] = ChatGoogleGenerativeAI(model=key[1], temperature=0)
        return _model_cache[key]
    else:
        key = _cache_key("ollama", "gemma4:12b-mlx")
        if key not in _model_cache:
            _model_cache[key] = ChatOllama(model="gemma4:12b-mlx", temperature=0.0)
        return _model_cache[key]


# --- Graph Wrapper Nodes ---

def local_orchestrator_node(state: AgentState):
    current_tools = get_all_tools()
    return call_orchestrator(state, model=get_model(), tools=list(current_tools.values()))


def local_agent_node(state: AgentState):
    return call_agent_node(state)


def local_responder_node(state: AgentState):
    return call_responder_node(state)


def local_tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    updates = {}

    current_tools = get_all_tools()

    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "tool_call",
        "details": "",
        "tool_calls": [],
    }

    # Depth tracks nesting *level*, not delegation count: the parent keeps
    # its own depth unchanged; each child is invoked with depth + 1.
    recursion_depth = state.get("recursion_depth", 0)
    child_usage = {"input": 0, "output": 0}
    child_writes: list[dict] = []

    # Collect task calls to check for parallel execution opportunity (4.2).
    task_calls = []
    non_task_calls = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name == "task":
            task_calls.append(tool_call)
        else:
            non_task_calls.append(tool_call)

    # Execute non-task tools first (these are sequential).
    for tool_call in non_task_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        audit_entry["tool_calls"].append({"name": tool_name, "args": tool_args})

        if tool_name in current_tools:
            tool_func = current_tools[tool_name]
            # Shared retry wrapper (exponential backoff); returns an error
            # string on failure rather than raising, so the graph can continue.
            try:
                result = invoke_with_retry(tool_func, tool_args)
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"

            if result is not None:
                if tool_name == "write_todos":
                    updates["current_plan"] = tool_args.get("todos", [])

                # Track write operations in pending_writes for audit/logging.
                # Writes execute immediately; pending_writes is informational only.
                if tool_name in ["write_file", "edit_file"]:
                    pending_entry = {
                        "tool": tool_name,
                        "tool_id": tool_id,
                        "args": tool_args,
                        "status": "executed",
                    }
                    existing_pending = state.get("pending_writes", [])
                    updates["pending_writes"] = existing_pending + [pending_entry]
        else:
            result = f"Tool '{tool_name}' not found."

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name))

    # 4.2: Execute task calls — run independent tasks in parallel when possible.
    # Tasks are considered independent if they have different subagent types
    # (research vs writer) or are general-purpose tasks with distinct descriptions.
    tasks_to_parallelize = []
    for tc in task_calls:
        tool_args = tc["args"]
        audit_entry["tool_calls"].append({"name": "task", "args": tool_args})

        if recursion_depth >= get_max_subagent_depth():
            result = (
                f"Error: Maximum subagent depth ({get_max_subagent_depth()}) reached. "
                f"Cannot delegate further subagents. "
                f"Handle this task directly or consolidate remaining work."
            )
            tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name="task"))
        else:
            subagent_type = tool_args.get("subagent_type", "general-purpose")
            description = tool_args.get("description", "")
            tasks_to_parallelize.append({
                "subagent_type": subagent_type,
                "description": description,
                "tool_id": tc["id"],
            })

    # Group tasks by the registry's parallelizable flag (6.1): those types run
    # concurrently; everything else (general-purpose, unknown types) runs
    # sequentially.
    independent_tasks = [t for t in tasks_to_parallelize
                        if is_parallelizable(t["subagent_type"])]
    sequential_tasks = [t for t in tasks_to_parallelize
                       if not is_parallelizable(t["subagent_type"])]

    def _run_task(task_info):
        result, usage, write_ops = _execute_task(
            task_info["subagent_type"],
            task_info["description"],
            recursion_depth,
            current_tools,
        )
        return (task_info["tool_id"], result, usage, write_ops)

    # One shared wall-clock deadline for the whole batch (6.6): each future
    # only gets the remaining time, so a hung batch costs one timeout, not
    # N x timeout.
    batch_deadline = time.monotonic() + get_subagent_timeout_seconds()

    def _resolve_task(future, task_info):
        """Collect one task result with the shared batch deadline. A failed or
        hung task yields an error string instead of dropping its siblings'
        results."""
        try:
            return future.result(timeout=max(0.0, batch_deadline - time.monotonic()))
        except Exception as e:
            return (task_info["tool_id"], f"Error executing subagent task: {e}", {}, [])

    def _record_task_result(tool_id, result, usage, write_ops):
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id, name="task"))
        child_usage["input"] += usage.get("input", 0)
        child_usage["output"] += usage.get("output", 0)
        child_writes.extend(write_ops)

    # Independent tasks: bounded parallel pool, results collected in
    # submission order so ToolMessage ordering is deterministic.
    if independent_tasks:
        executor = ThreadPoolExecutor(
            max_workers=min(get_max_parallel_tasks(), len(independent_tasks))
        )
        try:
            futures = [executor.submit(_run_task, t) for t in independent_tasks]
            for task_info, future in zip(independent_tasks, futures):
                _record_task_result(*_resolve_task(future, task_info))
        finally:
            # wait=False: a hung worker must not block the parent past the timeout.
            executor.shutdown(wait=False, cancel_futures=True)

    # Sequential tasks: one at a time (shared state), same timeout guard.
    for task_info in sequential_tasks:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_run_task, task_info)
            _record_task_result(*_resolve_task(future, task_info))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # Fold subagent token spend into the parent's budget so top-level
    # tracking reflects the whole delegation tree, not just this level.
    if child_usage["input"] or child_usage["output"]:
        current = state.get("token_usage", {}) or {}
        total_in = current.get("input", 0) + child_usage["input"]
        total_out = current.get("output", 0) + child_usage["output"]
        updates["token_usage"] = {
            "input": total_in,
            "output": total_out,
            "total": total_in + total_out,
        }

    # Subagent file writes join the parent's audit trail (previously only
    # top-level write_file/edit_file calls were tracked).
    if child_writes:
        existing_pending = state.get("pending_writes", [])
        updates["pending_writes"] = existing_pending + [
            {
                "tool": op.get("tool", "write_file"),
                "tool_id": op.get("tool_id"),
                "args": op.get("args", {}),
                "status": op.get("status", "executed"),
            }
            for op in child_writes
        ]

    updates["messages"] = tool_messages
    updates["workspace_files"] = get_workspace_files()
    updates["audit_log"] = [audit_entry]
    return updates


def local_reflection_node(state: AgentState):
    """Wrapper for the reflection node — prompts agent to reconsider its approach."""
    return call_reflection_node(state, get_model())


# --- Graph Construction ---

def get_deep_agent():
    """Return the compiled deep agent graph (cached after first call)."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    workspace_root = os.getenv("WORKSPACE_ROOT", "./workspace")
    if not os.path.exists(workspace_root):
        os.makedirs(workspace_root)

    skills_root = "./skills"
    if not os.path.exists(skills_root):
        os.makedirs(skills_root)

    workflow = StateGraph(AgentState)

    # --- Core nodes ---
    workflow.add_node("orchestrator", local_orchestrator_node)
    workflow.add_node("agent", local_agent_node)
    workflow.add_node("responder", local_responder_node)
    workflow.add_node("tools", local_tools_node)
    workflow.add_node("critic", lambda state: call_critic_node(state, get_model()))
    workflow.add_node("plan_checker", lambda state: call_plan_checker_node(state, get_model()))

    # --- 4.1: Reflection node — agent reconsiders approach before retrying ---
    workflow.add_node("reflection", local_reflection_node)

    workflow.set_entry_point("orchestrator")

    # Orchestrator routes to agent/critic/plan_checker/responder/end based on state.
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "agent": "agent",
            "critic": "critic",
            "plan_checker": "plan_checker",
            "responder": "responder",
            "end": END,
        },
    )

    # Agent (staging) → tools execution → back to orchestrator
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "orchestrator")

    # Critic routes: approved → responder, rejected → reflection or orchestrator.
    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "responder": "responder",
            "reflection": "reflection",
            "orchestrator": "orchestrator",
        },
    )

    # Plan checker routes: compliant → critic, violation → orchestrator.
    workflow.add_conditional_edges(
        "plan_checker",
        route_from_plan_checker,
        {
            "critic": "critic",
            "orchestrator": "orchestrator",
        },
    )

    # 4.1: Reflection → orchestrator (with revised strategy).
    workflow.add_conditional_edges(
        "reflection",
        route_from_reflection,
        {
            "orchestrator": "orchestrator",
        },
    )

    # Responder delivers the final answer.
    workflow.add_edge("responder", END)

    # --- Compile with checkpointer. Observability (8.3) is attached at the
    # model level via _maybe_attach_callbacks, so no per-node hooking here. ---
    checkpointer = MemorySaver()
    _compiled_graph = workflow.compile(checkpointer=checkpointer)
    return _compiled_graph


def reset_deep_agent():
    """Invalidate the cached compiled graph, model clients, and MCP tools.

    Call this when tools, skills, system prompts, or model configuration
    change and you need a fresh graph. The next call to ``get_deep_agent()``
    will compile a new instance and rebuild provider clients (8.5) and MCP
    tools (10.1).
    """
    global _compiled_graph
    _compiled_graph = None
    _model_cache.clear()
    clear_mcp_tools_cache()
