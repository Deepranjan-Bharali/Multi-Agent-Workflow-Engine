# app/core/state.py
from typing import TypedDict, Annotated, List
from typing_extensions import NotRequired
from langchain_core.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    current_plan: str
    documents: List[str]
    final_answer: str
    operation: NotRequired[str]
    source_document: NotRequired[str]
    document_chunks: NotRequired[List[str]]
