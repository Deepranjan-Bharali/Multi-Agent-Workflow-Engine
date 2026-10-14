# FastAPI and Celery workflow engine

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Start the API and a Celery worker in separate terminals from the repository root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
celery -A app.worker:celery_app worker --loglevel=INFO
```

The API persists task records with SQLAlchemy. Set `SQLALCHEMY_DATABASE_URL` to a shared PostgreSQL or SQLite SQLAlchemy async URL; it falls back to a local `agentic_flow_tasks.db` SQLite file. `DATABASE_URL` is also used as the task database URL when `SQLALCHEMY_DATABASE_URL` is unset. For multi-host deployments use PostgreSQL, not local SQLite.

Celery uses `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND`; both default to `REDIS_URL`, then `redis://localhost:6379/0`. LangGraph checkpointing is optional and enabled when `LANGGRAPH_DATABASE_URL` or `DATABASE_URL` is a PostgreSQL URL. The PostgreSQL connection should be reachable by both API and worker.

Workflow submission returns `202 Accepted` and a task ID. Poll `GET /api/v1/workflows/tasks/{task_id}` to read persistent status, logs, and result. The worker writes node progress to the task record as the graph runs.
