from typing import Annotated, List, TypedDict, Optional, Dict
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]  # append-only message log
    next_message: Optional[BaseMessage]           # staging area for LLM response
    pending_writes: List[Dict]                    # file write operations for audit
    audit_log: Annotated[List[Dict], add]         # append-only action log
    token_usage: Dict[str, int]                   # tracking API consumption
    enhanced_query: str                           # LLM-enhanced query
    department_targets: List[str]                 # departments detected by orchestrator
    subagent_results: Dict[str, str]              # results from parallel sub-agent dispatch
    # `recursion_depth`, `consecutive_invalid_tools`, `iteration_count`, and
    # `max_iterations` are deliberately absent. They belonged to the removed
    # top-level ReAct dispatcher loop: the graph is acyclic
    # (orchestrator → fanout → responder), so the router runs once per turn
    # and has nothing to count, and a checkpointed counter carried across
    # turns on a reused thread — turn 2 of every conversation dropped its
    # departments. Iteration counting lives where iteration happens, inside a
    # sub-agent's ReAct tool loop (`subagents.run_tool_loop`), bounded by its
    # own `max_iterations`. Per-turn values do not belong in checkpointed
    # state, and no parent state should carry a counter it never reads.
    #
    # `current_plan`, `workspace_files`, `routing_decisions`, `review_verdict`,
    # and `thread_id` were removed with them: no graph node reads or writes
    # any of them. The thread id lives in the runtime config.
