# app/main.py
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# 1. LOAD ENV VARS FIRST
load_dotenv()

# 2. THEN IMPORT APP MODULES
from app.api.routes import router as workflow_router
from app.core.memory import get_checkpointer, pool

# The lifespan manager runs code before the server accepts requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB and create LangGraph checkpoint tables
    checkpointer = await get_checkpointer()
    await checkpointer.setup()
    yield
    # Clean up the connection pool when the server shuts down
    if pool:
        await pool.close()

app = FastAPI(
    title="Agentic-Flow Engine", 
    lifespan=lifespan
)

app.include_router(workflow_router, prefix="/api/v1/workflows", tags=["Workflows"])