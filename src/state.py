from typing import Annotated, List, TypedDict, Optional
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    current_plan: List[str]
    workspace_files: List[str]
    next_message: Optional[BaseMessage]
    subagent_role: Optional[str]
