# app/core/memory.py
import os
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# We will initialize this pool from the FastAPI lifespan event
pool = None

async def get_async_pool():
    global pool
    if pool is None:
        db_uri = os.getenv("DATABASE_URL")
        # max_size=20 ensures we can handle concurrent agent runs
        pool = AsyncConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True})
        await pool.open()
    return pool

async def get_checkpointer():
    connection_pool = await get_async_pool()
    return AsyncPostgresSaver(connection_pool)

async def log_workflow_start(thread_id: str, project_id: int, prompt: str) -> int:
    """Inserts a 'running' record into Django's WorkflowRunLog table and returns the row ID."""
    pool = await get_async_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Django automatically names tables as _
            await cur.execute(
                """
                INSERT INTO workflows_workflowrunlog 
                (thread_id, project_id, prompt, status, created_at) 
                VALUES (%s, %s, %s, 'running', NOW()) 
                RETURNING id;
                """,
                (thread_id, project_id, prompt)
            )
            log_id = (await cur.fetchone())[0]
            return log_id

async def log_workflow_complete(log_id: int, final_answer: str, status: str = 'success'):
    """Updates the log record when the workflow finishes or fails."""
    pool = await get_async_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE workflows_workflowrunlog 
                SET final_answer = %s, status = %s, completed_at = NOW() 
                WHERE id = %s;
                """,
                (final_answer, status, log_id)
            )