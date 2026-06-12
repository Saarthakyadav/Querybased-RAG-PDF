import os
import chromadb
import numpy as np
from typing import List, Any
import hashlib

from config import (
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
    STORE_BATCH_SIZE,
)

class VectorStore:
    def __init__(
        self,
         collection_name:  str = COLLECTION_NAME,   
        persist_directory: str = VECTOR_STORE_DIR,
        batch_size:        int = STORE_BATCH_SIZE,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.batch_size = batch_size

        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize persistent ChromaDB with cosine similarity"""
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # ensures correct similarity
        )

        print(f"Vector store ready. Current count: {self.collection.count()}")

    # collision-safe ID generation
    def _generate_stable_id(self, doc, chunk_idx: int)->str:
        """
        Generate truly unique + stable ID using SHA256
        """

        content_hash = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()

        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "0")

        return f"{source}_{page}_{chunk_idx}_{content_hash[:16]}"

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add or update documents using batching + upsert
        """

        if not documents or len(embeddings) == 0:
            print("No documents to add.")
            return

        if len(documents) != len(embeddings):
            raise ValueError("Mismatch between documents and embeddings")

        total = len(documents)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size,total)

            batch_docs = documents[start:end]
            batch_embs = embeddings[start:end]

            ids, metadatas, texts, embeddings_list = [], [], [], []

            for local_idx, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                global_idx= start+ local_idx
                # UNIQUE + STABLE ID
                doc_id = self._generate_stable_id(doc,global_idx)
                ids.append(doc_id)

                # Clean metadata
                clean_meta = {}

                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif v is not None:
                        clean_meta[k] = str(v)

                # Add useful metadata
                clean_meta.update({
                    "chunk_id": global_idx,
                    "content_length": len(doc.page_content),
                    "type": doc.metadata.get("type", "text")
                })

                metadatas.append(clean_meta)
                texts.append(doc.page_content)

                embeddings_list.append(
                    emb.tolist() if hasattr(emb, "tolist") else emb
                )

            #  UPSERT (no duplicates)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=texts
            )

            print(f"Inserted batch {start} → {min(end, total)}")

        print(f"Added/Updated {total} documents.")

    def reset(self):
        """Delete the collection and recreate it (used by the UI clear button)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store reset.")