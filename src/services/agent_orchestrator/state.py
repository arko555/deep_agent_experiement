from typing import Annotated, List, TypedDict, Optional, Dict
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]  # append-only message log
    current_plan: List[str]                       # todo list (updated by write_todos)
    workspace_files: List[str]                    # synced after every tool execution
    next_message: Optional[BaseMessage]           # staging area for LLM response
    review_verdict: Optional[str]                 # critic / plan_checker verdict
    recursion_depth: int                          # subagent nesting level
    pending_writes: List[Dict]                    # file write operations for audit
    audit_log: Annotated[List[Dict], add]         # append-only action log
    routing_decisions: List[Dict]                 # department routing history
    token_usage: Dict[str, int]                   # tracking API consumption
    iteration_count: int                          # current loop count
    max_iterations: int                           # limit on loops
