from typing import Annotated, List, TypedDict, Optional, Dict
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]  # append-only message log
    current_plan: List[str]                       # todo list (updated by write_todos)
    workspace_files: List[str]                    # synced after every tool execution
    next_message: Optional[BaseMessage]           # staging area for LLM response
    review_verdict: Optional[str]                 # critic / plan_checker verdict
    pending_writes: List[Dict]                    # file write operations for audit
    audit_log: Annotated[List[Dict], add]         # append-only action log
    routing_decisions: List[Dict]                 # department routing history
    token_usage: Dict[str, int]                   # tracking API consumption
    # Phase 3 fields
    enhanced_query: str                           # LLM-enhanced query
    department_targets: List[str]                 # departments detected by orchestrator
    subagent_results: Dict[str, str]              # results from parallel sub-agent dispatch
    thread_id: str                                # conversation thread (from runtime config)
    # `recursion_depth` and `consecutive_invalid_tools` were removed with the
    # top-level tools node. There is no sub-agent nesting left to bound — a
    # sub-agent's tool loop is bounded by its own iteration budget — and no
    # graph-level loop for an unknown-tool counter to cut short.
    #
    # `iteration_count` and `max_iterations` were removed here too. They
    # belonged to the top-level ReAct dispatcher loop, which no longer exists:
    # the graph is acyclic (orchestrator → fanout → responder), so the router
    # runs once per turn. Worse, they were checkpointed, so the counter
    # carried across turns on a reused thread and every conversation after the
    # first dropped its departments. Iteration counting belongs at the layer
    # that actually iterates — inside a sub-agent's ReAct tool loop, where
    # `subagents.run_tool_loop` bounds turns with its own `max_iterations`
    # argument. No parent state should carry a counter it never reads.

