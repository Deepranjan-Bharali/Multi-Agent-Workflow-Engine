# seed_db.py
from dotenv import load_dotenv
load_dotenv()

#from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings # New Import
from langchain_core.documents import Document

# Initialize the EXACT SAME Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = PineconeVectorStore(index_name="agentic-flow", embedding=embeddings)

doc = Document(page_content="Deepranjan's secret Agentic-Flow project code name is 'Operation Crimson'.")

# vectorstore.add_documents([doc])
print("Successfully added document to Pinecone using Hugging Face embeddings!")