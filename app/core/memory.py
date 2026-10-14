"""Optional PostgreSQL checkpointer for LangGraph conversation state."""
import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

pool: AsyncConnectionPool | None = None


def checkpointing_enabled() -> bool:
    url = os.getenv("LANGGRAPH_DATABASE_URL") or os.getenv("DATABASE_URL", "")
    return url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://"))


async def get_async_pool() -> AsyncConnectionPool:
    global pool
    if pool is None:
        db_uri = os.getenv("LANGGRAPH_DATABASE_URL") or os.getenv("DATABASE_URL")
        if not checkpointing_enabled() or not db_uri:
            raise RuntimeError("A PostgreSQL LANGGRAPH_DATABASE_URL is required to enable checkpointing")
        db_uri = db_uri.replace("postgresql+psycopg://", "postgresql://", 1)
        pool = AsyncConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True})
        await pool.open()
    return pool


async def get_checkpointer() -> AsyncPostgresSaver:
    return AsyncPostgresSaver(await get_async_pool())


async def close_resources() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None
