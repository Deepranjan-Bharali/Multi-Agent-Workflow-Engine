"""Celery task runner for LangGraph workflows."""
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from celery import Celery
from langchain_core.messages import HumanMessage

from app.core.agents.text_utils import split_document
from app.core.graph import get_compiled_graph
from app.core.memory import checkpointing_enabled, close_resources, get_checkpointer
from app.db import append_task_log, dispose_db, get_task_input, set_task_status

logger = logging.getLogger(__name__)
BROKER_URL = (
    os.getenv("CELERY_BROKER_URL")
    or os.getenv("REDIS_URL")
    or "redis://localhost:6379/0"
)
RESULT_BACKEND = (
    os.getenv("CELERY_RESULT_BACKEND")
    or os.getenv("REDIS_URL")
    or "redis://localhost:6379/1"
)

celery_app = Celery("agentic_flow", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_timeout=3,
    broker_connection_retry_on_startup=True,
    task_publish_retry=False,
)

NODE_LABELS = {
    "planner": "Planning document analysis",
    "retriever": "Retrieving relevant external context",
    "executor": "Analyzing and consolidating document sections",
}


async def _execute_workflow(task_id: str) -> None:
    task_input = await get_task_input(task_id)
    if task_input is None:
        logger.error("Workflow task %s does not exist in the task database", task_id)
        return

    await set_task_status(task_id, "running")
    try:
        chunks = split_document(task_input["prompt"])
        await append_task_log(task_id, f"Split source into {len(chunks)} analysis section(s).")

        checkpointer = None
        if checkpointing_enabled():
            checkpointer = await get_checkpointer()
            await checkpointer.setup()
        graph = get_compiled_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": task_input["thread_id"]}}
        initial_state = {
            "messages": [HumanMessage(content=f"Operation: {task_input['operation']}. Analyze the complete source document from the supplied sections.")],
            "operation": task_input["operation"],
            "document_chunks": chunks,
        }

        plan, answer, total_messages = None, "", 1
        async for update in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in update.items():
                await append_task_log(task_id, NODE_LABELS.get(node_name, f"Completed {node_name}"))
                plan = node_state.get("current_plan", plan)
                answer = node_state.get("final_answer", answer)
                total_messages += len(node_state.get("messages", []))

        result = {"plan": plan, "answer": answer, "total_messages": total_messages, "cache": "miss"}
        await set_task_status(task_id, "success", result=result)
    except Exception as exc:
        logger.exception("Workflow task %s failed", task_id)
        await append_task_log(task_id, f"Workflow failed: {exc}")
        await set_task_status(task_id, "failed", error=str(exc))
        raise
    finally:
        # Celery runs each task in its own asyncio loop; close loop-bound pools
        # before asyncio.run tears the loop down.
        try:
            await close_resources()
        finally:
            await dispose_db()


@celery_app.task(name="app.worker.execute_workflow")
def execute_workflow(task_id: str) -> dict[str, str]:
    """Execute one persisted workflow; the HTTP API only enqueues this task."""
    asyncio.run(_execute_workflow(task_id))
    return {"task_id": task_id, "status": "finished"}
