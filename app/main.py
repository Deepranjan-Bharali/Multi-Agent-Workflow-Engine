# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os

# 1. LOAD ENV VARS FIRST
load_dotenv()

# 2. THEN IMPORT APP MODULES
from app.api.routes import router as workflow_router
from app.core.memory import checkpointing_enabled, close_resources, get_checkpointer
from app.db import dispose_db, init_db

# The lifespan manager runs code before the server accepts requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        # LangGraph's checkpoint saver remains optional; SQLAlchemy stores task state.
        if checkpointing_enabled():
            checkpointer = await get_checkpointer()
            await checkpointer.setup()
        yield
    finally:
        await close_resources()
        await dispose_db()

app = FastAPI(
    title="Agentic-Flow AI Engine",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3001").split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(workflow_router, prefix="/api/v1/workflows", tags=["Workflows"])
