# ============================================================
# src/retriever.py
# Hybrid retrieval: BM25 (sparse) + ChromaDB (dense) + Cross-encoder reranking
# ============================================================

from __future__ import annotations

import math
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """
    Combines BM25 (keyword) and dense (ChromaDB) retrieval using
    Reciprocal Rank Fusion, then reranks with a cross-encoder.

    Drop-in replacement for the raw ChromaDB query in RAGSystem.
    """

    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        vector_store,
        embedding_manager,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.use_reranker = use_reranker

        # Cross-encoder loaded once
        self._reranker: Optional[CrossEncoder] = None
        if use_reranker:
            try:
                self._reranker = CrossEncoder(self.RERANKER_MODEL)
                print(f"[HybridRetriever] Cross-encoder loaded: {self.RERANKER_MODEL}")
            except Exception as e:
                print(f"[HybridRetriever] Reranker unavailable ({e}), skipping.")

        # BM25 index — rebuilt each time retrieve() is called because
        # ChromaDB documents can change between sessions.
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_all_docs(self) -> List[Dict[str, Any]]:
        """Pull every document out of ChromaDB for BM25 indexing."""
        count = self.vector_store.collection.count()
        if count == 0:
            return []

        result = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )
        docs = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            docs.append({"content": text, "metadata": meta})
        return docs

    def _build_bm25(self, docs: List[Dict[str, Any]]) -> BM25Okapi:
        tokenized = [d["content"].lower().split() for d in docs]
        return BM25Okapi(tokenized)

    def _dense_retrieve(
        self, query: str, top_n: int, score_threshold: float, filter_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Return top-n dense results above threshold."""
        embeddings, _ = self.embedding_manager.generate_embeddings([query])
        if len(embeddings) == 0:
            return []
        query_emb = embeddings[0]

        where_clause = {"type": filter_type.lower()} if filter_type else None

        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_n, self.vector_store.collection.count() or top_n),
            where=where_clause,
        )

        docs = []
        if results.get("documents") and results["documents"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                score = max(0.0, 1.0 - (distance / 2))
                if score >= score_threshold:
                    docs.append({
                        "content":  results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score":    round(score, 4),
                        "id":       results["ids"][0][i],
                    })
        return sorted(docs, key=lambda x: x["score"], reverse=True)

    def _bm25_retrieve(
        self, query: str, top_n: int
    ) -> List[Dict[str, Any]]:
        """Return top-n BM25 results with normalised scores."""
        if not self._bm25 or not self._bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_n]
        max_score = scores[top_indices[0]] if len(top_indices) > 0 else 1.0

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append({
                **self._bm25_docs[idx],
                "score": float(scores[idx] / max_score) if max_score > 0 else 0.0,
                "id":    f"bm25_{idx}",
            })
        return results

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Dict[str, Any]],
        bm25_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists via RRF.
        Uses content as dedup key (ids may differ between sources).
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_docs):
            key = doc["content"][:200]  # content prefix as key
            scores[key] = scores.get(key, 0.0) + self.dense_weight * self._rrf_score(rank, self.rrf_k)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc["content"][:200]
            scores[key] = scores.get(key, 0.0) + self.bm25_weight * self._rrf_score(rank, self.rrf_k)
            if key not in doc_map:
                doc_map[key] = doc

        merged = []
        for key, fused_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = dict(doc_map[key])
            doc["hybrid_score"] = round(fused_score, 6)
            merged.append(doc)

        return merged

    def _rerank(
        self, query: str, docs: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking on merged candidates."""
        if not self._reranker or not docs:
            return docs[:top_k]

        pairs = [(query, d["content"]) for d in docs]
        try:
            ce_scores = self._reranker.predict(pairs)
            for doc, score in zip(docs, ce_scores):
                doc["rerank_score"] = float(score)
            docs = sorted(docs, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        except Exception as e:
            print(f"[HybridRetriever] Reranking failed ({e}), using hybrid scores.")

        return docs[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        filter_type: Optional[str] = None,
        candidate_multiplier: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline:
        1. Dense retrieval (ChromaDB)
        2. BM25 retrieval (in-memory index over all stored docs)
        3. RRF fusion
        4. Cross-encoder reranking
        5. Return top_k
        """
        candidate_n = top_k * candidate_multiplier

        # --- Dense ---
        dense_docs = self._dense_retrieve(query, candidate_n, score_threshold, filter_type)

        # --- BM25: rebuild index if needed ---
        all_docs = self._fetch_all_docs()
        if all_docs:
            self._bm25_docs = all_docs
            self._bm25 = self._build_bm25(all_docs)
            bm25_docs = self._bm25_retrieve(query, candidate_n)
        else:
            bm25_docs = []

        # --- Fallback: if one source is empty ---
        if not dense_docs and not bm25_docs:
            return []
        if not bm25_docs:
            fused = dense_docs
        elif not dense_docs:
            fused = bm25_docs
        else:
            fused = self._reciprocal_rank_fusion(dense_docs, bm25_docs)

        # --- Rerank ---
        reranked = self._rerank(query, fused, top_k)

        # Normalise final score field for downstream consumers
        for doc in reranked:
            doc["score"] = doc.get("rerank_score") or doc.get("hybrid_score") or doc.get("score", 0.0)

        return reranked
