from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import torch

from config import EMBED_MODEL, EMBED_BATCH_SIZE
 


class EmbeddingManager:
    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH_SIZE):
        """
        Production-ready embedding manager
        """

        self.model_name = model_name
        self.batch_size = batch_size

        #  Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def _clean_text(self, text: str) -> str:
        """
        Light normalization (preserve semantics)
        """
        if not text:
            return ""

        # Remove excessive whitespace only
        text = " ".join(text.strip().split())
        return text

    def _safe_truncate(self, text: str, max_words: int = 400) -> str:
        """
        Prevent silent truncation by transformer models
        """
        words = text.split()
        return " ".join(words[:max_words])

    def generate_embeddings(self, texts: List[str])->Tuple[np.ndarray,List[int]]:

        """
        Returns
        -------
        embeddings   : np.ndarray  shape (N, dim)
        valid_indices: List[int]   indices into the original `texts` list
                                   that were actually embedded (non-empty).
        """
        processed_texts: List[str] = []
        valid_indices:   List[int] = []

        for i, t in enumerate(texts):
            if not t and not t.strip():
                continue
            cleaned   = self._clean_text(t)
            truncated = self._safe_truncate(cleaned)
            processed_texts.append(truncated)
            valid_indices.append(i)

        if not processed_texts:
            return np.array([]), []

        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return np.array(embeddings), valid_indices