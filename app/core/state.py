# app/core/state.py
from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    current_plan: str
    documents: List[str]
    final_answer: str