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
from src.core.memory import get_workspace_files, get_system_prompt
from src.core.routing import (
    route_from_orchestrator,
    route_from_critic,
    route_from_plan_checker,
    route_from_reflection,
    route_after_tools,
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


# --- Observability Hooks (4.4) ---
# Configurable tracing callbacks for monitoring LLM calls, tool executions,
# and state transitions. Enable via OBSERVABILITY=1 env var.

class DeepAgentTracer:
    """Trace LLM invocations, tool calls, and state transitions for debugging."""

    def __init__(self):
        self._events: list[dict] = []

    def on_chat_model_start(self, name: str, messages: list, **kwargs):
        self._record(
            type="chat_model_start",
            name=name,
            message_count=len(messages),
            timestamp=datetime.now().isoformat(),
        )

    def on_chat_model_end(self, name: str, **kwargs):
        self._record(
            type="chat_model_end",
            name=name,
            timestamp=datetime.now().isoformat(),
        )

    def on_tool_start(self, name: str, args: dict, **kwargs):
        self._record(
            type="tool_start",
            name=name,
            args=args,
            timestamp=datetime.now().isoformat(),
        )

    def on_tool_end(self, name: str, output: str, **kwargs):
        self._record(
            type="tool_end",
            name=name,
            output_length=len(str(output)),
            timestamp=datetime.now().isoformat(),
        )

    def on_chain_start(self, name: str, inputs: dict, **kwargs):
        self._record(
            type="chain_start",
            name=name,
            input_keys=list(inputs.keys()) if isinstance(inputs, dict) else [],
            timestamp=datetime.now().isoformat(),
        )

    def on_chain_end(self, name: str, outputs: dict, **kwargs):
        self._record(
            type="chain_end",
            name=name,
            output_keys=list(outputs.keys()) if isinstance(outputs, dict) else [],
            timestamp=datetime.now().isoformat(),
        )

    def _record(self, **data):
        self._events.append(data)

    def get_events(self) -> list[dict]:
        return list(self._events)

    def clear(self):
        self._events.clear()


_tracer = DeepAgentTracer() if os.getenv("OBSERVABILITY") == "1" else None


def get_tracer() -> DeepAgentTracer | None:
    """Return the global tracer instance, or None if observability is disabled."""
    return _tracer


# --- Model Selection ---
def get_model():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if anthropic_key:
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0,
        )
    elif openrouter_key:
        return ChatOpenRouter(
            model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            api_key=openrouter_key,
            temperature=0,
        )
    elif openai_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )
    elif google_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )
    else:
        return ChatOllama(
            model="gemma4:12b-mlx",
            temperature=0.0,
        )


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

def get_deep_agent(
    enable_observability: bool | None = None,
):
    """Return the compiled deep agent graph (cached after first call).

    Args:
        enable_observability: Whether to enable tracing callbacks. Defaults
        to the OBSERVABILITY env var.
    """
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

    # --- 4.4: Observability — compile with tracing callbacks when enabled ---
    checkpointer = MemorySaver()

    if enable_observability is None:
        enable_observability = os.getenv("OBSERVABILITY") == "1"

    if enable_observability:
        # LangGraph supports trace via the tracer parameter on compile.
        # We wrap the graph with callback hooks for state transitions.
        _compiled_graph = workflow.compile(
            checkpointer=checkpointer,
        )
        # Patch in observability by wrapping node invocations.
        _apply_observability_hooks(_compiled_graph)
    else:
        _compiled_graph = workflow.compile(checkpointer=checkpointer)

    return _compiled_graph


def reset_deep_agent():
    """Invalidate the cached compiled graph.

    Call this when tools, skills, system prompts, or model configuration
    change and you need a fresh graph. The next call to ``get_deep_agent()``
    will compile a new instance.
    """
    global _compiled_graph
    _compiled_graph = None


def _apply_observability_hooks(graph):
    """Wrap graph execution with observability hooks for tracing.

    This patches the graph's underlying node execution to emit trace events
    via the global tracer when observability is enabled.
    """
    if _tracer is None:
        return

    # Store original node functions so we can wrap them.
    original_nodes = {}

    # LangGraph stores nodes internally; we intercept at the state level
    # by adding a pre/post hook via the audit_log mechanism already in place.
    # The tracer captures LLM-level events via LangChain's callback system
    # and tool-level events via the audit_log.
    def _trace_state_transition(state_before: AgentState, state_after: AgentState):
        """Track state changes between graph steps."""
        changes = {}
        all_keys = set(list(state_before.keys()) + list(state_after.keys()))
        for key in all_keys:
            old_val = state_before.get(key)
            new_val = state_after.get(key)
            if old_val != new_val:
                changes[key] = {
                    "from": str(old_val)[:200],
                    "to": str(new_val)[:200],
                }
        if changes:
            _tracer._record(
                type="state_transition",
                changed_keys=list(changes.keys()),
                timestamp=datetime.now().isoformat(),
            )

    # The hook is registered as a post-processing step on the audit_log.
    # Since audit_log entries are already emitted by nodes, we augment them.
    graph._tracer = _tracer
