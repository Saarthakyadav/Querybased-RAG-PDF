# ============================================================
# src/rag_system.py
# ============================================================

import logging
import time
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage

from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        vector_store,
        embedding_manager,
        llm,
        use_hybrid: bool = True,
        use_reranker: bool = True,
    ):
        self.vector_store      = vector_store
        self.embedding_manager = embedding_manager
        self.llm               = llm
        self.use_hybrid        = use_hybrid

        self._hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            use_reranker=use_reranker,
        )

        # FIX: wire the vector store's post-ingest callback to the retriever's
        # cache invalidation so BM25 is rebuilt on next query after any ingestion.
        self.vector_store._on_documents_added = self._hybrid_retriever.invalidate_bm25_cache

    # ------------------------------------------------------------------
    # Legacy helper — kept for backward compatibility
    # ------------------------------------------------------------------
    def _convert_distance_to_score(self, distance: float) -> float:
        return max(0.0, 1.0 - (distance / 2))

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if self.use_hybrid:
            return self._hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_type=filter_type,
            )

        # ---- Fallback: original dense-only path ----
        embeddings, _ = self.embedding_manager.generate_embeddings([query])
        if len(embeddings) == 0:
            return []
        query_emb = embeddings[0]

        collection_size = self.vector_store.collection.count()
        if collection_size == 0:
            return []

        where_clause = {"type": filter_type.lower()} if filter_type else None
        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k * 3, collection_size),
            where=where_clause,
        )

        retrieved_docs: List[Dict[str, Any]] = []
        if results.get("documents") and results["documents"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                score    = self._convert_distance_to_score(distance)
                if score >= score_threshold:
                    retrieved_docs.append({
                        "content":  results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score":    round(score, 4),
                    })

        return sorted(retrieved_docs, key=lambda x: x["score"], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------
    def ask(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ) -> Dict[str, Any]:

        start_time = time.time()
        docs       = self.retrieve(query, top_k, score_threshold)
        fetch_time = time.time() - start_time

        if not docs:
            return {
                "answer":  "Not found in document",
                "sources": [],
                "chunks":  [],
                "metrics": {},
            }

        context = "\n\n".join(
            f"[Source {i+1} | Page {d['metadata'].get('page','?')} | File: {d['metadata'].get('source_file','?')}]\n{d['content'][:1200]}"
            for i, d in enumerate(docs)
        )

        prompt = f"""
You are an expert research paper assistant.

Your job is to give a thorough, detailed answer using ONLY the provided context.

Instructions:
- Give a COMPLETE and DETAILED answer — do not cut it short.
- Cover ALL relevant points found across all sources.
- After every key point cite like this: (Page X, filename).
- If the same idea appears in multiple sources, mention all of them.
- Use bullet points or numbered lists for multi-part questions.
- Include specific values, methods, findings, and conclusions from the text.
- Do NOT use any knowledge outside the provided context.
- If the answer is genuinely not present, say: "Not found in document"

Context:
{context}

Question: {query}

Answer:
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer   = response.content.strip()
        if len(answer) < 10:
            answer = "Not found in document"

        # FIX: report the score of the top result, not an average across
        # incompatible score scales (rerank / hybrid / cosine).
        top_score = docs[0]["score"] if docs else 0.0

        return {
            "answer":  answer,
            "sources": [d["metadata"] for d in docs],
            "chunks":  docs,
            "metrics": {
                "fetch_time":      f"{fetch_time:.2f}s",
                "top_chunk_score": f"{top_score * 100:.1f}%",
            },
        }