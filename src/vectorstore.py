import os
import hashlib
import logging
import chromadb
import numpy as np
from typing import List, Any, Optional

from config import (
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
    STORE_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        collection_name:   str = COLLECTION_NAME,
        persist_directory: str = VECTOR_STORE_DIR,
        batch_size:        int = STORE_BATCH_SIZE,
    ):
        self.collection_name  = collection_name
        self.persist_directory = persist_directory
        self.batch_size        = batch_size

        # Optional callback — set by RAGSystem so the BM25 cache is
        # invalidated automatically after every successful ingestion.
        self._on_documents_added: Optional[callable] = None

        self.client     = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store ready. Current count: %d", self.collection.count())

    def _generate_stable_id(self, doc, chunk_idx: int) -> str:
        """
        Generate a truly stable, collision-safe ID.
        FIX: the original used `chunk_idx` (loop position) as part of the key,
        which changed depending on how many other documents were ingested at the
        same time, causing duplicate insertions instead of upserts.
        New key uses only content hash + source + page — always the same for the
        same chunk regardless of batch order.
        """
        content_hash = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()
        source = doc.metadata.get("source_file", "unknown")
        page   = doc.metadata.get("page", "0")
        return f"{source}__p{page}__{content_hash[:24]}"

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if not documents or len(embeddings) == 0:
            logger.info("No documents to add.")
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
            )

        total = len(documents)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_docs = documents[start:end]
            batch_embs = embeddings[start:end]

            ids, metadatas, texts, embeddings_list = [], [], [], []

            for local_idx, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                global_idx = start + local_idx
                doc_id     = self._generate_stable_id(doc, global_idx)
                ids.append(doc_id)

                clean_meta = {}
                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif v is not None:
                        clean_meta[k] = str(v)

                clean_meta.update({
                    "chunk_index":    global_idx,
                    "content_length": len(doc.page_content),
                    "type":           doc.metadata.get("type", "text"),
                })

                metadatas.append(clean_meta)
                texts.append(doc.page_content)
                embeddings_list.append(
                    emb.tolist() if hasattr(emb, "tolist") else emb
                )

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=texts,
            )
            logger.info("Upserted batch %d → %d", start, end)

        logger.info("Added/Updated %d documents.", total)

        # Notify the retriever so it can rebuild the BM25 index on next query.
        if self._on_documents_added:
            self._on_documents_added()

    def reset(self):
        """Delete the collection and recreate it."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset.")

        if self._on_documents_added:
            self._on_documents_added()