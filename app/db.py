"""SQLAlchemy persistence for asynchronous workflow tasks."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, func
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _database_url() -> str:
    url = os.getenv("SQLALCHEMY_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        db_file = (Path(__file__).resolve().parent.parent / "agentic_flow_tasks.db").as_posix()
        return f"sqlite+aiosqlite:///{db_file}"
    parsed = make_url(url)
    if parsed.drivername in {"postgres", "postgresql", "postgresql+psycopg"}:
        query = dict(parsed.query)
        if "sslmode" in query:
            query["ssl"] = query.pop("sslmode")
        parsed = parsed.set(drivername="postgresql+asyncpg", query=query)
    return parsed.render_as_string(hide_password=False)


DATABASE_URL = _database_url()
_engine_options: dict[str, Any] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    # Keep SQLite's connection usable across the async adapter's worker thread.
    _engine_options["connect_args"] = {"check_same_thread": False}
else:
    _engine_options.update(pool_size=int(os.getenv("DB_POOL_SIZE", "10")), max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")))

engine = create_async_engine(DATABASE_URL, **_engine_options)
SessionFactory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class WorkflowTask(Base):
    __tablename__ = "workflow_tasks"
    __table_args__ = (Index("ix_workflow_tasks_status_created", "status", "created_at"),)

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    project_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(40), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    logs: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False, default=list)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


async def init_db() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    await engine.dispose()


async def create_task(task_id: str, thread_id: str, project_id: int, operation: str, prompt: str) -> None:
    async with SessionFactory() as session:
        session.add(WorkflowTask(
            task_id=task_id,
            thread_id=thread_id,
            project_id=project_id,
            operation=operation,
            prompt=prompt,
            status="queued",
            logs=[{"at": datetime.now(timezone.utc).isoformat(), "message": "Task accepted and queued."}],
        ))
        await session.commit()


async def get_task(task_id: str) -> dict[str, Any] | None:
    async with SessionFactory() as session:
        task = await session.get(WorkflowTask, task_id)
        return _serialize_task(task) if task else None


async def get_task_input(task_id: str) -> dict[str, Any] | None:
    async with SessionFactory() as session:
        task = await session.get(WorkflowTask, task_id)
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "thread_id": task.thread_id,
            "project_id": task.project_id,
            "operation": task.operation,
            "prompt": task.prompt,
        }


async def append_task_log(task_id: str, message: str) -> None:
    async with SessionFactory() as session:
        task = await session.get(WorkflowTask, task_id)
        if task is None:
            return
        task.logs = [*task.logs, {"at": datetime.now(timezone.utc).isoformat(), "message": message}]
        await session.commit()


async def set_task_status(task_id: str, status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    async with SessionFactory() as session:
        task = await session.get(WorkflowTask, task_id)
        if task is None:
            return
        task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if status in {"success", "failed"}:
            task.completed_at = datetime.now(timezone.utc)
        await session.commit()


def _serialize_task(task: WorkflowTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "thread_id": task.thread_id,
        "project_id": task.project_id,
        "operation": task.operation,
        "status": task.status,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "logs": task.logs or [],
        "result": task.result,
        "error": task.error,
    }
