"""Async FastAPI gateway with Redis-backed semantic response caching."""
import asyncio
import hashlib
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("agentic_flow.gateway")

ENGINE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_CONNECTIONS = int(os.getenv("GATEWAY_MAX_CONNECTIONS", "500"))
CACHE_SCAN_BATCH_SIZE = max(1, int(os.getenv("CACHE_SCAN_BATCH_SIZE", "128")))

http: httpx.AsyncClient | None = None
redis: Any = None
embeddings: Any = None
embedding_lock = asyncio.Lock()
task_requests: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http, redis
    http = httpx.AsyncClient(
        timeout=httpx.Timeout(20.0, connect=3.0),
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=100),
    )
    try:
        from redis.asyncio import Redis

        redis = Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=1, socket_timeout=2)
        await redis.ping()
    except Exception:
        logger.exception("Redis client initialization failed; semantic caching is disabled")
        if redis is not None:
            try:
                await redis.aclose()
            except Exception:
                logger.debug("Could not close unavailable Redis connection", exc_info=True)
        redis = None
    yield
    await http.aclose()
    if redis is not None:
        await redis.aclose()


app = FastAPI(title="Agentic Flow API Gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3001").split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


async def get_embedder():
    global embeddings
    if embeddings is None:
        async with embedding_lock:
            if embeddings is None:
                def load_model():
                    from langchain_huggingface import HuggingFaceEmbeddings

                    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

                embeddings = await asyncio.to_thread(load_model)
    return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def cache_query(payload: dict[str, Any]) -> str:
    operation = payload.get("operation", "technical_summary")
    return f"Operation: {operation}\n\n{payload.get('prompt', '')}"


def cache_prefix(payload: dict[str, Any]) -> str:
    return f"gateway-semantic-cache:{payload.get('project_id', 'default')}:"


async def find_cached_response(payload: dict[str, Any]) -> dict[str, Any] | None:
    if redis is None or not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        return None
    try:
        embedder = await get_embedder()
        query_vector = await asyncio.to_thread(embedder.embed_query, cache_query(payload))
        best_score, best_result = 0.0, None
        batch: list[str] = []

        async def score_batch(keys: list[str]) -> None:
            nonlocal best_score, best_result
            values = await redis.mget(*keys)
            for raw in values:
                if not raw:
                    continue
                item = json.loads(raw)
                score = cosine_similarity(query_vector, item["embedding"])
                if score > best_score:
                    best_score, best_result = score, item.get("result")

        async for key in redis.scan_iter(match=f"{cache_prefix(payload)}*", count=100):
            batch.append(key)
            if len(batch) >= CACHE_SCAN_BATCH_SIZE:
                await score_batch(batch)
                batch = []
        if batch:
            await score_batch(batch)
        if best_score > CACHE_THRESHOLD and best_result is not None:
            logger.info("Semantic cache hit (similarity=%.4f)", best_score)
            return best_result
    except Exception:
        logger.exception("Semantic cache lookup failed; forwarding request to engine")
    return None


async def store_cached_response(payload: dict[str, Any], result: dict[str, Any]) -> None:
    if redis is None or not result.get("answer"):
        return
    try:
        embedder = await get_embedder()
        query = cache_query(payload)
        vector = await asyncio.to_thread(embedder.embed_query, query)
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        key = f"{cache_prefix(payload)}{digest}"
        await redis.set(key, json.dumps({"embedding": vector, "result": result}), ex=CACHE_TTL_SECONDS)
    except Exception:
        logger.exception("Semantic cache write failed")


@app.get("/health")
async def health():
    return {"status": "ok", "semantic_cache": "enabled" if redis is not None else "unavailable"}


@app.post("/api/v1/workflows/execute")
async def execute_workflow(request: Request):
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise HTTPException(status_code=422, detail="Request must include a non-empty prompt")

    cached = await find_cached_response(payload)
    if cached is not None:
        return {"status": "success", "cached": True, "data": cached}

    try:
        upstream = await http.post(f"{ENGINE_URL}/api/v1/workflows/execute", json=payload)
    except httpx.RequestError as exc:
        logger.warning("AI engine request failed: %s", exc)
        raise HTTPException(status_code=502, detail="AI engine is unavailable") from exc
    if upstream.status_code == 202:
        try:
            accepted = upstream.json()
            task_id = accepted.get("task_id")
            if task_id:
                if len(task_requests) >= 10_000:
                    task_requests.pop(next(iter(task_requests)))
                task_requests[task_id] = payload
                if redis is not None:
                    await redis.set(f"gateway-task:{task_id}", json.dumps(payload), ex=CACHE_TTL_SECONDS * 24)
        except Exception:
            logger.exception("Could not retain submitted task metadata for cache population")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


@app.get("/api/v1/workflows/tasks/{task_id}")
async def get_workflow_task(task_id: str):
    try:
        upstream = await http.get(f"{ENGINE_URL}/api/v1/workflows/tasks/{task_id}")
    except httpx.RequestError as exc:
        logger.warning("AI engine task lookup failed: %s", exc)
        raise HTTPException(status_code=502, detail="AI engine is unavailable") from exc

    if upstream.is_success:
        try:
            task = upstream.json()
            if task.get("status") == "success" and isinstance(task.get("result"), dict):
                payload = task_requests.get(task_id)
                if payload is None and redis is not None:
                    raw_payload = await redis.get(f"gateway-task:{task_id}")
                    payload = json.loads(raw_payload) if raw_payload else None
                if payload is not None:
                    await store_cached_response(payload, task["result"])
                    task_requests.pop(task_id, None)
                    if redis is not None:
                        await redis.delete(f"gateway-task:{task_id}")
        except Exception:
            logger.warning("Could not inspect upstream task response for cache storage")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))
