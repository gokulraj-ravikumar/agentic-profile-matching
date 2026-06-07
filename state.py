from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    job_description: str
    requirements: Dict[str, Any]
    candidate_pool: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]]
    current_stage: str
    feedback: str
