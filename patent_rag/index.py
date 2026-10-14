"""Loading and validating the precomputed vector index (built by notebooks 02 and 03).

Index = two aligned files:
  * embeddings/embeddings.npy        -> float32 matrix, one 384-dim row per chunk
  * embeddings/chunk_metadata.jsonl  -> one JSON record per chunk, same row order
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import IndexIntegrityError, IndexNotFoundError

REQUIRED_CHUNK_FIELDS = ("id", "patent_id", "chunk_index", "text")


@dataclass
class Chunk:
    id: str
    patent_id: str
    chunk_index: int
    text: str


@dataclass
class VectorIndex:
    """Chunk records plus their L2-normalized embedding matrix (row i <-> chunks[i])."""

    chunks: list[Chunk]
    embeddings: np.ndarray

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def patent_ids(self) -> list[str]:
        return sorted({c.patent_id for c in self.chunks})


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so cosine similarity becomes a plain dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise IndexNotFoundError(detail=f"missing chunk metadata file: {path}")
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IndexIntegrityError(detail=f"{path.name} line {line_no}: invalid JSON") from exc
            missing = [k for k in REQUIRED_CHUNK_FIELDS if k not in rec]
            if missing:
                raise IndexIntegrityError(detail=f"{path.name} line {line_no}: missing fields {missing}")
            chunks.append(
                Chunk(
                    id=str(rec["id"]),
                    patent_id=str(rec["patent_id"]),
                    chunk_index=int(rec["chunk_index"]),
                    text=str(rec["text"]),
                )
            )
    if not chunks:
        raise IndexIntegrityError(detail=f"{path.name} contains no chunks")
    return chunks


def load_index(embeddings_path: Path, metadata_path: Path) -> VectorIndex:
    """Load and validate the index. Raises user-safe errors on missing/corrupt files."""
    if not embeddings_path.is_file():
        raise IndexNotFoundError(detail=f"missing embeddings file: {embeddings_path}")
    chunks = load_chunks(metadata_path)
    try:
        embeddings = np.load(embeddings_path)
    except (ValueError, OSError) as exc:
        raise IndexIntegrityError(detail=f"could not read {embeddings_path.name}") from exc

    if embeddings.ndim != 2:
        raise IndexIntegrityError(detail=f"embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(chunks):
        raise IndexIntegrityError(
            detail=f"{embeddings.shape[0]} embeddings but {len(chunks)} chunk records"
        )
    if not np.isfinite(embeddings).all():
        raise IndexIntegrityError(detail="embeddings contain NaN or inf values")

    return VectorIndex(chunks=chunks, embeddings=normalize_rows(embeddings.astype(np.float32)))
