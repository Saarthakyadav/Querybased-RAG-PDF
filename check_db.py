from src.vectorstore import VectorStore
import chromadb

store = VectorStore()
# Check how many chunks are actually in your database
count = store.collection.count()
print(f"✅ Database Status: {count} chunks found.")

# Peek at the first 2 entries to see if metadata (author, type) is correct
if count > 0:
    peek = store.collection.peek(limit=2)
    print("🔍 Sample Metadata:", peek['metadatas'][0])