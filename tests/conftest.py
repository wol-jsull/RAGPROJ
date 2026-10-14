"""Shared fixtures: a tiny in-memory index, fake embedder and fake LLM (no network, no model download)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patent_rag.config import Settings  # noqa: E402
from patent_rag.index import Chunk, VectorIndex, normalize_rows  # noqa: E402
from patent_rag.metadata import PatentCatalog  # noqa: E402
from patent_rag.pipeline import RAGPipeline  # noqa: E402
from patent_rag.retrieval import SemanticRetriever  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Three "topics" -> axis-aligned 4-dim vectors so similarity scores are predictable.
TOPIC_VECTORS = {
    "attention": [1.0, 0.0, 0.0, 0.0],
    "dialogue": [0.0, 1.0, 0.0, 0.0],
    "battery": [0.0, 0.0, 1.0, 0.0],
}


class KeywordFakeEmbedder:
    """Maps a query to a topic vector by keyword; unknown queries -> orthogonal 4th axis."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            t = t.lower()
            vec = [0.0, 0.0, 0.0, 1.0]
            for topic, v in TOPIC_VECTORS.items():
                if topic in t:
                    vec = v
                    break
            out.append(vec)
        return np.array(out, dtype=np.float32)


class FakeLLM:
    def __init__(self, reply="**From the patents:**\n- Attention is computed sparsely [S1].", error=None):
        self.model = "fake-model"
        self.reply = reply
        self.error = error
        self.calls: list = []

    def complete(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def tiny_index() -> VectorIndex:
    chunks = [
        Chunk("PA_0", "PA", 0, "Sparse attention mechanism for transformers."),
        Chunk("PA_1", "PA", 1, "Attention heads attend to a subset of positions."),
        Chunk("PB_0", "PB", 0, "Multi-turn dialogue history is encoded with BERT."),
        Chunk("PC_0", "PC", 0, "Heat-radiating fluid for a battery module."),
    ]
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return VectorIndex(chunks=chunks, embeddings=normalize_rows(emb))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(Settings(), demo_cache_path=tmp_path / "recorded.json", min_similarity=0.3)


@pytest.fixture
def catalog() -> PatentCatalog:
    return PatentCatalog(
        {
            "PA": {"number": "US 1", "title": "Sparse Attention", "applicant": "A Corp", "type": "Granted patent"},
            "PB": {"number": "US 2", "title": "Dialogue Transformer"},
        }
    )


@pytest.fixture
def make_pipeline(tiny_index, settings, catalog):
    def _make(llm=None, recorded=None, **overrides):
        s = replace(settings, **overrides) if overrides else settings
        retrievers = {"semantic": SemanticRetriever(tiny_index, KeywordFakeEmbedder())}
        return RAGPipeline(s, retrievers, catalog, llm=llm, recorded=recorded)

    return _make
