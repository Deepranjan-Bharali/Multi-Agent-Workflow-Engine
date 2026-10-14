"""Retrieve supporting context from Pinecone when configured."""
import asyncio
import logging
import os

from app.core.agents.text_utils import representative_text
from app.core.state import AgentState

logger = logging.getLogger(__name__)
_vector_store = None
_store_lock = asyncio.Lock()


async def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        async with _store_lock:
            if _vector_store is None:
                def create_store():
                    from langchain_huggingface import HuggingFaceEmbeddings
                    from langchain_pinecone import PineconeVectorStore

                    embedding = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
                    return PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-flow"), embedding=embedding)

                _vector_store = await asyncio.to_thread(create_store)
    return _vector_store


async def retriever_node(state: AgentState):
    if not os.getenv("PINECONE_API_KEY"):
        return {"documents": []}
    chunks = state.get("document_chunks", [])
    query = f"{state.get('current_plan', '')}\n\n{representative_text(chunks, max_chars=5000)}"
    if not query.strip():
        return {"documents": []}
    try:
        store = await _get_vector_store()
        docs = await store.asimilarity_search(query, k=4)
        return {"documents": [doc.page_content for doc in docs if doc.page_content]}
    except Exception:
        logger.exception("Pinecone retrieval failed; continuing without external context")
        return {"documents": []}
