"""
rag_service.py
==============
FIX: The original code called build_rag_runtime() on every API request,
re-loading the 90MB embedding model and reconnecting to ChromaDB each time.
This module now initialises the runtime ONCE at import time (module-level
singleton) so every request reuses the same objects.

FIX: ingest_pdfs() now only processes the newly uploaded files, not every
PDF already in PDF_DIR, eliminating duplicate-chunk accumulation.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# ---------------------------------------------------------------------------
# Module-level singleton — built ONCE when the module is first imported.
# All API handlers share this instance.
# ---------------------------------------------------------------------------

def _build_runtime() -> Tuple[RAGSystem, ImageProcessor]:
    logger.info("[rag_service] Initialising RAG runtime …")
    llm        = ChatGroq(model=LLM_MODEL)
    embeddings = EmbeddingManager(batch_size=EMBED_BATCH_SIZE)
    store      = VectorStore(batch_size=STORE_BATCH_SIZE)
    image_proc = ImageProcessor()
    rag        = RAGSystem(
        store,
        embeddings,
        llm,
        use_hybrid=USE_HYBRID_RETRIEVAL,
        use_reranker=USE_RERANKER,
    )
    logger.info("[rag_service] Runtime ready.")
    return rag, image_proc


_rag_instance:         RAGSystem    = None
_image_proc_instance:  ImageProcessor = None


def _get_runtime() -> Tuple[RAGSystem, ImageProcessor]:
    """Return the singleton runtime, building it on first call."""
    global _rag_instance, _image_proc_instance
    if _rag_instance is None:
        _rag_instance, _image_proc_instance = _build_runtime()
    return _rag_instance, _image_proc_instance


def get_rag() -> RAGSystem:
    rag, _ = _get_runtime()
    return rag


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def ingest_pdfs(uploaded_files: List[Any]) -> Dict[str, Any]:
    """
    Save uploaded files and ingest ONLY the new files.
    FIX: original re-processed every PDF in PDF_DIR on every upload,
    causing duplicates and wasted embedding work.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
    rag, image_processor = _get_runtime()

    # Save new files and remember their paths.
    saved_paths: List[str] = []
    for uploaded_file in uploaded_files:
        dest = os.path.join(PDF_DIR, uploaded_file.filename)
        with open(dest, "wb") as f:
            f.write(uploaded_file.file.read())
        saved_paths.append(dest)

    if not saved_paths:
        return {"status": "success", "files_processed": 0, "chunks_created": 0, "duration_seconds": 0}

    processor = PDFProcessor(
        llm=rag.llm,
        vision_func=image_processor.get_image_description,
        process_images=True,
    )

    # Create a temporary directory with ONLY the newly uploaded files so
    # process_pdfs() doesn't re-process files that are already indexed.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmp_dir:
        for path in saved_paths:
            shutil.copy(path, tmp_dir)

        start_total = time.time()
        raw_docs    = processor.process_pdfs(tmp_dir)
        chunks      = processor.split_documents(raw_docs)
        contents    = [doc.page_content for doc in chunks]
        embeddings_arr, valid_indices = rag.embedding_manager.generate_embeddings(contents)
        chunks = [chunks[i] for i in valid_indices]

        if len(chunks) != len(embeddings_arr):
            raise ValueError("Chunk and embedding count mismatch.")

        rag.vector_store.add_documents(chunks, embeddings_arr)

    return {
        "status":           "success",
        "files_processed":  len(uploaded_files),
        "chunks_created":   len(chunks),
        "duration_seconds": round(time.time() - start_total, 2),
    }


def query_documents(
    question:       str,
    top_k:          int   = TOP_K_DEFAULT,
    score_threshold: float = SCORE_THRESHOLD,
    use_hybrid:     bool  = USE_HYBRID_RETRIEVAL,
    use_reranker:   bool  = USE_RERANKER,
) -> Dict[str, Any]:
    rag, _ = _get_runtime()
    # Update toggles on the shared instance (no reconstruction needed).
    rag.use_hybrid = use_hybrid
    rag._hybrid_retriever.use_reranker = use_reranker
    if not use_reranker:
        rag._hybrid_retriever._reranker = None
    return rag.ask(question, top_k=top_k, score_threshold=score_threshold)


def clear_documents() -> Dict[str, Any]:
    import shutil
    shutil.rmtree(PDF_DIR, ignore_errors=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    rag, _ = _get_runtime()
    rag.vector_store.reset()

    return {"status": "cleared"}