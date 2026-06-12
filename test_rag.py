from src.search import RAGSystem
from src.vectorstore import VectorStore
from src.embedding import EmbeddingManager
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize everything
store = VectorStore()
embeddings = EmbeddingManager()
llm = ChatGroq(model="llama-3.3-70b-versatile")
rag = RAGSystem(store, embeddings, llm)

# Run a sample query
response = rag.ask("explain The Transformer - model architecture diagram layer by layer")
print("\n AI Answer:\n", response['answer'])
print("\n Sources Used:", [s['source_file'] for s in response['sources']])