# ============================================================
# src/retriever.py
# Hybrid retrieval: BM25 (sparse) + ChromaDB (dense) + Cross-encoder reranking
# ============================================================

from __future__ import annotations

import hashlib
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Maximum number of candidates fed into the cross-encoder to keep latency bounded.
_MAX_RERANK_CANDIDATES = 20


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

        # Cross-encoder loaded once at startup.
        self._reranker: Optional[CrossEncoder] = None
        if use_reranker:
            try:
                self._reranker = CrossEncoder(self.RERANKER_MODEL)
                logger.info("[HybridRetriever] Cross-encoder loaded: %s", self.RERANKER_MODEL)
            except Exception as e:
                logger.warning("[HybridRetriever] Reranker unavailable (%s), skipping.", e)

        # BM25 index — built lazily on first query, then invalidated only when
        # the collection size changes (i.e., after ingestion).
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict[str, Any]] = []
        self._bm25_doc_count: int = 0   # tracks when a rebuild is needed

    # ------------------------------------------------------------------
    # BM25 cache management
    # ------------------------------------------------------------------

    def invalidate_bm25_cache(self) -> None:
        """Call this after new documents are ingested to force a BM25 rebuild."""
        self._bm25 = None
        self._bm25_docs = []
        self._bm25_doc_count = 0
        logger.info("[HybridRetriever] BM25 cache invalidated.")

    def _ensure_bm25_index(self) -> None:
        """
        Build (or rebuild) the BM25 index only when the collection size has
        changed since the last build.  For very large collections the fetch is
        still expensive once, but after that every query reuses the cached index.
        """
        current_count = self.vector_store.collection.count()
        if current_count == 0:
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_doc_count = 0
            return

        if self._bm25 is not None and current_count == self._bm25_doc_count:
            # Index is still valid — nothing to do.
            return

        logger.info(
            "[HybridRetriever] Building BM25 index over %d documents …", current_count
        )
        # FIX: fetch in pages to avoid a single giant memory allocation.
        # ChromaDB's get() supports limit + offset.
        PAGE = 2000
        all_docs: List[Dict[str, Any]] = []
        offset = 0
        while True:
            result = self.vector_store.collection.get(
                include=["documents", "metadatas"],
                limit=PAGE,
                offset=offset,
            )
            batch = result.get("documents") or []
            if not batch:
                break
            for text, meta in zip(batch, result["metadatas"]):
                all_docs.append({"content": text, "metadata": meta})
            if len(batch) < PAGE:
                break
            offset += PAGE

        tokenized = [d["content"].lower().split() for d in all_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = all_docs
        self._bm25_doc_count = current_count
        logger.info("[HybridRetriever] BM25 index built (%d docs).", len(all_docs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_retrieve(
        self, query: str, top_n: int, score_threshold: float, filter_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Return top-n dense results above threshold."""
        embeddings, _ = self.embedding_manager.generate_embeddings([query])
        if len(embeddings) == 0:
            return []
        query_emb = embeddings[0]

        where_clause = {"type": filter_type.lower()} if filter_type else None
        collection_size = self.vector_store.collection.count()
        if collection_size == 0:
            return []

        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_n, collection_size),
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

    def _bm25_retrieve(self, query: str, top_n: int) -> List[Dict[str, Any]]:
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
        FIX: use a stable SHA-256 content hash as the dedup key instead of a
        fragile 200-char prefix (which caused silent collisions).
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        def _key(doc: Dict[str, Any]) -> str:
            return hashlib.sha256(doc["content"].encode("utf-8")).hexdigest()

        for rank, doc in enumerate(dense_docs):
            key = _key(doc)
            scores[key] = scores.get(key, 0.0) + self.dense_weight * self._rrf_score(rank, self.rrf_k)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = _key(doc)
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
        """
        Apply cross-encoder reranking on merged candidates.
        FIX: cap the input to _MAX_RERANK_CANDIDATES to bound latency.
        """
        if not self._reranker or not docs:
            return docs[:top_k]

        # Only rerank a bounded set; discard the rest before the expensive call.
        candidates = docs[:_MAX_RERANK_CANDIDATES]

        pairs = [(query, d["content"]) for d in candidates]
        try:
            ce_scores = self._reranker.predict(pairs)
            for doc, score in zip(candidates, ce_scores):
                doc["rerank_score"] = float(score)
            candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        except Exception as e:
            logger.warning("[HybridRetriever] Reranking failed (%s), using hybrid scores.", e)

        return candidates[:top_k]

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
        2. BM25 retrieval (cached in-memory index — rebuilt only on collection change)
        3. RRF fusion
        4. Cross-encoder reranking (capped at _MAX_RERANK_CANDIDATES)
        5. Return top_k
        """
        candidate_n = top_k * candidate_multiplier

        # --- Dense ---
        dense_docs = self._dense_retrieve(query, candidate_n, score_threshold, filter_type)

        # --- BM25: use cached index, rebuild only if collection changed ---
        self._ensure_bm25_index()
        bm25_docs = self._bm25_retrieve(query, candidate_n) if self._bm25 else []

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