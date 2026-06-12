from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import torch
import logging

from config import EMBED_MODEL, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 has a hard 256-token limit.
# At ~1.3 tokens/word, 190 words is a safe ceiling to avoid silent truncation.
_MAX_WORDS = 190


class EmbeddingManager:
    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def _clean_text(self, text: str) -> str:
        """Light normalisation — preserve semantics."""
        if not text:
            return ""
        return " ".join(text.strip().split())

    def _safe_truncate(self, text: str, max_words: int = _MAX_WORDS) -> str:
        """
        Truncate to max_words to prevent silent truncation by the transformer.
        FIX: default reduced from 400 to 190 to respect the 256-token model limit.
        """
        words = text.split()
        return " ".join(words[:max_words])

    def generate_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Returns
        -------
        embeddings   : np.ndarray  shape (N, dim)
        valid_indices: List[int]   indices of texts that were actually embedded.

        FIX: guards against None values (raises AttributeError in original code).
        FIX: show_progress_bar disabled — was printing tqdm bars on every query.
        """
        processed_texts: List[str] = []
        valid_indices:   List[int] = []

        for i, t in enumerate(texts):
            # FIX: original check `not t and not t.strip()` was dead code and
            # would crash on None. Correct: skip if falsy OR blank after strip.
            if not t or not str(t).strip():
                continue
            cleaned   = self._clean_text(str(t))
            truncated = self._safe_truncate(cleaned)
            processed_texts.append(truncated)
            valid_indices.append(i)

        if not processed_texts:
            return np.array([]), []

        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,   # FIX: was True — clutters stdout in production
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.array(embeddings), valid_indices