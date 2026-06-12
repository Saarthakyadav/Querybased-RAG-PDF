# ============================================================
# src/dataset_generator.py
# Generates synthetic (question, ground_truth, contexts) triples
# from already-ingested ChromaDB documents — no PDFs needed at eval time.
# ============================================================

from __future__ import annotations

import json
import math  # FIX: was incorrectly placed at the bottom of the file
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


_GENERATION_PROMPT = """You are creating a question-answer evaluation dataset for a RAG system.

Given the following document chunk from a research paper, generate {n} high-quality question-answer pairs.

Rules:
- Questions must be answerable ONLY from the provided text — no outside knowledge.
- Questions should be specific and varied: factual, conceptual, and comparative.
- Answers must be complete, detailed, and drawn directly from the text.
- Do NOT generate yes/no questions.
- Return ONLY a JSON array — no markdown, no explanation, no backticks.

Format:
[
  {{
    "question": "...",
    "ground_truth": "..."
  }}
]

Document chunk:
\"\"\"
{chunk}
\"\"\"
"""


class DatasetGenerator:
    """
    Pulls chunks from ChromaDB, samples them, and asks the LLM to generate
    Q&A pairs. Saves results to data/eval_dataset.json.
    """

    DEFAULT_SAVE_PATH = "data/eval_dataset.json"

    def __init__(self, vector_store, llm, rag_system=None):
        self.vector_store = vector_store
        self.llm          = llm
        self.rag_system   = rag_system  # used to retrieve contexts per question

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_chunks(self, max_chunks: int = 80) -> List[Dict[str, Any]]:
        """Pull documents from ChromaDB for sampling."""
        count = self.vector_store.collection.count()
        if count == 0:
            return []

        result = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )
        docs = [
            {"content": text, "metadata": meta}
            for text, meta in zip(result["documents"], result["metadatas"])
            # FIX: guard against None values from ChromaDB before calling .strip()
            if text and len(text.strip()) > 200
        ]
        # Shuffle and cap so we get variety across the corpus
        random.shuffle(docs)
        return docs[:max_chunks]

    def _generate_pairs_from_chunk(
        self, chunk: str, n: int = 2
    ) -> List[Dict[str, str]]:
        """Call LLM to generate n Q&A pairs from one chunk."""
        prompt = _GENERATION_PROMPT.format(chunk=chunk[:1500], n=n)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw      = response.content.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            pairs = json.loads(raw)
            if isinstance(pairs, list):
                return [
                    p for p in pairs
                    if isinstance(p, dict) and "question" in p and "ground_truth" in p
                ]
        except Exception as e:
            print(f"[DatasetGenerator] Skipping chunk — parse error: {e}")
        return []

    def _attach_contexts(
        self, pairs: List[Dict[str, Any]], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        For each Q&A pair, retrieve context chunks via the RAG system
        and attach them. These become the 'retrieved_contexts' field
        that RAGAS uses for context_precision / context_recall.
        """
        if not self.rag_system:
            return pairs

        enriched = []
        for pair in pairs:
            try:
                docs = self.rag_system.retrieve(
                    pair["question"], top_k=top_k, score_threshold=0.2
                )
                pair["contexts"] = [d["content"] for d in docs]
            except Exception as e:
                print(f"[DatasetGenerator] Context fetch failed: {e}")
                pair["contexts"] = []
            enriched.append(pair)
            time.sleep(0.3)   # light rate-limit guard

        return enriched

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        num_questions: int = 20,
        questions_per_chunk: int = 2,
        save_path: str = DEFAULT_SAVE_PATH,
        attach_contexts: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate `num_questions` synthetic Q&A pairs from stored documents.

        Returns a list of dicts:
            { question, ground_truth, contexts (list[str]) }
        Also saves to `save_path`.
        """
        chunks_needed = math.ceil(num_questions / questions_per_chunk)
        all_chunks    = self._fetch_chunks(max_chunks=chunks_needed * 2)

        if not all_chunks:
            raise RuntimeError(
                "No documents found in vector store. "
                "Please ingest PDFs before generating a dataset."
            )

        dataset: List[Dict[str, Any]] = []
        chunk_iter = iter(all_chunks)

        while len(dataset) < num_questions:
            try:
                chunk_doc = next(chunk_iter)
            except StopIteration:
                break

            pairs = self._generate_pairs_from_chunk(
                chunk_doc["content"], n=questions_per_chunk
            )
            # Tag each pair with source metadata
            for pair in pairs:
                pair["source_metadata"] = chunk_doc.get("metadata", {})

            dataset.extend(pairs)
            time.sleep(0.5)   # brief pacing for generation calls

        dataset = dataset[:num_questions]

        if attach_contexts and self.rag_system:
            dataset = self._attach_contexts(dataset, top_k=3)

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"[DatasetGenerator] Saved {len(dataset)} pairs → {save_path}")
        return dataset

    @staticmethod
    def load(path: str = DEFAULT_SAVE_PATH) -> List[Dict[str, Any]]:
        """Load a previously generated dataset from disk."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)