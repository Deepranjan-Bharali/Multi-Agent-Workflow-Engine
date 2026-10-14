"""HTTP endpoints for submitting and monitoring persisted Celery workflow tasks."""
import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException

from app.db import append_task_log, create_task, get_task, set_task_status
from app.schemas.requests import WorkflowRequest
from app.worker import execute_workflow

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/execute", status_code=202)
async def submit_workflow(request: WorkflowRequest):
    task_id = str(uuid.uuid4())
    await create_task(
        task_id=task_id,
        thread_id=request.thread_id,
        project_id=request.project_id,
        operation=request.operation,
        prompt=request.prompt,
    )
    try:
        await asyncio.to_thread(execute_workflow.delay, task_id)
    except Exception as exc:
        logger.exception("Could not enqueue workflow task %s", task_id)
        await append_task_log(task_id, "Could not enqueue task; check Celery broker availability.")
        await set_task_status(task_id, "failed", error="Task queue is unavailable")
        raise HTTPException(status_code=503, detail="Task queue is unavailable") from exc
    return {
        "task_id": task_id,
        "status": "accepted",
        "status_url": f"/api/v1/workflows/tasks/{task_id}",
    }


@router.get("/tasks/{task_id}")
async def read_workflow_task(task_id: str):
    task = await get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
