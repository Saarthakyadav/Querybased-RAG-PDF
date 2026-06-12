import logging
import os
import shutil
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import PDFProcessor
from src.embedding import EmbeddingManager
from src.image_processor import ImageProcessor
from src.rag_system import RAGSystem
from src.vectorstore import VectorStore

from config import (
    EMBED_BATCH_SIZE,
    LLM_MODEL,
    PDF_DIR,
    SCORE_THRESHOLD,
    STORE_BATCH_SIZE,
    TOP_K_DEFAULT,
    USE_HYBRID_RETRIEVAL,
    USE_RERANKER,
)

load_dotenv()
logger = logging.getLogger("docinsight.rag_service")


def _build_runtime() -> Tuple[RAGSystem, ImageProcessor]:
    llm = ChatGroq(model=LLM_MODEL)
    embeddings = EmbeddingManager(batch_size=EMBED_BATCH_SIZE)
    store = VectorStore(batch_size=STORE_BATCH_SIZE)
    image_proc = ImageProcessor()
    rag = RAGSystem(
        store,
        embeddings,
        llm,
        use_hybrid=USE_HYBRID_RETRIEVAL,
        use_reranker=USE_RERANKER,
    )
    return rag, image_proc


def build_rag_runtime() -> Tuple[RAGSystem, ImageProcessor]:
    return _build_runtime()


def ingest_pdfs(uploaded_files: List[Any]) -> Dict[str, Any]:
    os.makedirs(PDF_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        dest = os.path.join(PDF_DIR, uploaded_file.filename)
        with open(dest, "wb") as f:
            f.write(uploaded_file.file.read())

    rag, image_processor = build_rag_runtime()
    processor = PDFProcessor(
        llm=rag.llm,
        vision_func=image_processor.get_image_description,
        process_images=True,
    )

    start_total = time.time()
    raw_docs = processor.process_pdfs(PDF_DIR)
    chunks = processor.split_documents(raw_docs)
    contents = [doc.page_content for doc in chunks]
    embeddings_arr, valid_indices = rag.embedding_manager.generate_embeddings(contents)
    chunks = [chunks[i] for i in valid_indices]

    if len(chunks) != len(embeddings_arr):
        raise ValueError("Chunk and embedding count mismatch.")

    rag.vector_store.add_documents(chunks, embeddings_arr)

    return {
        "status": "success",
        "files_processed": len(uploaded_files),
        "chunks_created": len(chunks),
        "duration_seconds": round(time.time() - start_total, 2),
    }


def query_documents(question: str, top_k: int = TOP_K_DEFAULT, score_threshold: float = SCORE_THRESHOLD,
                    use_hybrid: bool = USE_HYBRID_RETRIEVAL, use_reranker: bool = USE_RERANKER) -> Dict[str, Any]:
    rag, _ = build_rag_runtime()
    rag.use_hybrid = use_hybrid
    rag._hybrid_retriever.use_reranker = use_reranker
    return rag.ask(question, top_k=top_k, score_threshold=score_threshold)


def clear_documents() -> Dict[str, Any]:
    shutil.rmtree(PDF_DIR, ignore_errors=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    rag, _ = build_rag_runtime()
    rag.vector_store.reset()

    return {"status": "cleared"}
