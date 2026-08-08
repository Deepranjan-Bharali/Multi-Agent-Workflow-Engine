# app/core/agents/retriever.py
import os
#from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings # New Import
from app.core.state import AgentState

# Initialize Hugging Face Embeddings
# all-MiniLM-L6-v2 is fast, lightweight, and perfect for local development
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-flow")

async def retriever_node(state: AgentState):
    messages = state.get("messages", [])
    user_prompt = messages[-1].content if messages else ""

    try:
        # vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        docs = await vectorstore.asimilarity_search(user_prompt, k=2)
        doc_contents = [doc.page_content for doc in docs]
    except Exception as e:
        print(f"Pinecone retrieval failed: {e}")
        doc_contents = ["No external documents could be retrieved."]

    return {"documents": doc_contents}