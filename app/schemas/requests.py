# app/schemas/requests.py
from pydantic import BaseModel
from typing import Literal

class WorkflowRequest(BaseModel):
    thread_id: str  # Required so LangGraph knows which state to load
    prompt: str
    project_id: int
    operation: Literal["entity_extraction", "technical_summary", "classification"] = "technical_summary"
