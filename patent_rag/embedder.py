"""Query embedding with the same SentenceTransformer model used to build the index."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .errors import EmbeddingModelError


class Embedder(Protocol):
    """Anything that turns a list of strings into a (n, dim) array. Tests use a fake."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Thin wrapper so the heavy import happens lazily and failures become user-safe errors."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(detail="sentence-transformers is not installed") from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # download/cache/IO failures surface as many exception types
            raise EmbeddingModelError(detail=f"failed to load {model_name}: {type(exc).__name__}") from exc

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
