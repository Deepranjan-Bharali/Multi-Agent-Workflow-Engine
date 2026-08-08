# app/schemas/requests.py
from pydantic import BaseModel
from typing import Optional

class WorkflowRequest(BaseModel):
    thread_id: str  # Required so LangGraph knows which state to load
    prompt: str
    project_id: int