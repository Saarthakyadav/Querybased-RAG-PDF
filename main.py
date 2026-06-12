import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import PDFProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.image_processor  import   ImageProcessor 

from config import (         
    LLM_MODEL,                
    EMBED_BATCH_SIZE,
    STORE_BATCH_SIZE,
    PDF_DIR,
)

load_dotenv()


def run_ingestion():
    start_total = time.time()

    # 1. Initialize Components
    llm = ChatGroq(model=LLM_MODEL)
    image_proc = ImageProcessor()


    # disable image processing for speed
    processor = PDFProcessor(
        llm=llm,
        vision_func=image_proc.get_image_description,
        process_images=True 
    )

    embedding_manager = EmbeddingManager(batch_size=EMBED_BATCH_SIZE)
    vector_store = VectorStore(batch_size=STORE_BATCH_SIZE)

    # 2. Extract
    print("\n🚀 Step 1: Extracting PDFs...")
    t1 = time.time()
    raw_docs = processor.process_pdfs("data/pdf_files")
    text_count  = sum(1 for d in raw_docs if d.metadata.get("type") == "text")
    image_count = sum(1 for d in raw_docs if d.metadata.get("type") == "image")
    print(f"✅ Extracted documents: {len(raw_docs)}")
    print(f"⏱️ Time: {time.time() - t1:.2f}s")

    # 3. Split
    print("\n✂️ Step 2: Splitting documents...")
    t2 = time.time()
    chunks = processor.split_documents(raw_docs)
    print(f"✅ Total chunks: {len(chunks)}")
    print(f"⏱️ Time: {time.time() - t2:.2f}s")

    # 4. Prepare texts
    print("\n📝 Step 3: Preparing texts...")
    texts = [doc.page_content for doc in chunks]
    print(f"✅ Texts ready: {len(texts)}")

    # 5. Embed (BATCH)
    print("\n🧠 Step 4: Generating embeddings...")
    t3 = time.time()
    embeddings,valid_indices = embedding_manager.generate_embeddings(texts)
    chunks = [chunks[i] for i in valid_indices]
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"⏱️ Time: {time.time() - t3:.2f}s")

    
    embeddings, valid_indices = embedding_manager.generate_embeddings(texts)
    chunks = [chunks[i] for i in valid_indices]
    print(f"✅ Embeddings shape: {embeddings.shape}  ⏱ {time.time()-t3:.2f}s")
 
    # This assert will now always pass
    assert len(chunks) == len(embeddings), "❌ Mismatch between chunks and embeddings!"
    # 6. Store (BATCH)
    print("\n💾 Step 5: Storing in vector DB...")
    t4 = time.time()
    vector_store.add_documents(chunks, embeddings)
    print(f"⏱️ Time: {time.time() - t4:.2f}s")

    print("\n🎉 Ingestion Complete!")
    print(f"⏱️ Total Time: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    llm = ChatGroq(model=LLM_MODEL)
    run_ingestion()