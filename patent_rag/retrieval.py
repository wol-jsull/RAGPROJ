"""Retrieval: semantic (embedding cosine similarity) and a keyword (TF-IDF) baseline.

The original notebooks used ``sklearn.neighbors.NearestNeighbors(metric="cosine")``,
which does an exact brute-force search. Because the embeddings are L2-normalized,
the same exact search is a single matrix-vector product; ``tests/test_retrieval.py``
checks that both give identical rankings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .embedder import Embedder
from .index import VectorIndex, normalize_rows


@dataclass
class RetrievedChunk:
    rank: int  # 1-based
    score: float  # cosine similarity (semantic) or TF-IDF cosine (keyword)
    chunk_id: str
    patent_id: str
    chunk_index: int
    text: str


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    method: str  # "semantic" or "keyword"
    latency_ms: float

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    # argpartition is O(n); then sort just the k winners. Stable tie-break on index.
    candidate = np.argpartition(-scores, k - 1)[:k]
    return candidate[np.lexsort((candidate, -scores[candidate]))]


def _mask_for(index: VectorIndex, patent_filter: Iterable[str] | None) -> np.ndarray | None:
    if not patent_filter:
        return None
    allowed = set(patent_filter)
    return np.array([c.patent_id in allowed for c in index.chunks])


def _package(index: VectorIndex, scores: np.ndarray, order: np.ndarray) -> list[RetrievedChunk]:
    out = []
    for rank, i in enumerate(order, start=1):
        c = index.chunks[int(i)]
        out.append(
            RetrievedChunk(
                rank=rank,
                score=float(scores[int(i)]),
                chunk_id=c.id,
                patent_id=c.patent_id,
                chunk_index=c.chunk_index,
                text=c.text,
            )
        )
    return out


class SemanticRetriever:
    method = "semantic"

    def __init__(self, index: VectorIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.asarray(self.embedder.encode([query]), dtype=np.float32)
        if vec.ndim != 2 or vec.shape[1] != self.index.dimension:
            raise ValueError(
                f"query embedding has shape {vec.shape}, index dimension is {self.index.dimension}"
            )
        return normalize_rows(vec)[0]

    def search_vector(
        self, query_vec: np.ndarray, top_k: int, patent_filter: Iterable[str] | None = None
    ) -> list[RetrievedChunk]:
        scores = self.index.embeddings @ query_vec  # cosine similarity per chunk
        mask = _mask_for(self.index, patent_filter)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            top_k = min(top_k, int(mask.sum()))
        return _package(self.index, scores, _top_k_indices(scores, top_k))

    def search(
        self, query: str, top_k: int, patent_filter: Iterable[str] | None = None
    ) -> RetrievalResult:
        start = time.perf_counter()
        chunks = self.search_vector(self.embed_query(query), top_k, patent_filter)
        return RetrievalResult(chunks, self.method, (time.perf_counter() - start) * 1000)


class KeywordRetriever:
    """TF-IDF keyword baseline, used to contrast exact-term matching with semantic search."""

    method = "keyword"

    def __init__(self, index: VectorIndex):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.index = index
        self._vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
        self._matrix = self._vectorizer.fit_transform([c.text for c in index.chunks])

    def search(
        self, query: str, top_k: int, patent_filter: Iterable[str] | None = None
    ) -> RetrievalResult:
        start = time.perf_counter()
        q = self._vectorizer.transform([query])
        scores = (self._matrix @ q.T).toarray().ravel()  # rows are L2-normalized -> cosine
        mask = _mask_for(self.index, patent_filter)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            top_k = min(top_k, int(mask.sum()))
        chunks = _package(self.index, scores, _top_k_indices(scores, top_k))
        return RetrievalResult(chunks, self.method, (time.perf_counter() - start) * 1000)
