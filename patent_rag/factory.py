"""Wires the real components together (used by the app and scripts)."""

from __future__ import annotations

from .config import Settings
from .embedder import SentenceTransformerEmbedder
from .errors import IndexIntegrityError, LLMConfigurationError
from .index import load_index
from .llm import OpenAIChatModel
from .metadata import PatentCatalog
from .pipeline import RAGPipeline
from .recorded import RecordedAnswers
from .retrieval import KeywordRetriever, SemanticRetriever


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Load index + models. Raises user-safe PatentRAGError subclasses on failure.

    A missing API key is NOT fatal: the pipeline runs in retrieval-only mode.
    """
    index = load_index(settings.embeddings_path, settings.chunk_metadata_path)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    if embedder.dimension != index.dimension:
        raise IndexIntegrityError(
            f"The embedding model ({settings.embedding_model}, {embedder.dimension}-dim) does not "
            f"match the index ({index.dimension}-dim). Use the model the index was built with.",
        )
    retrievers = {
        "semantic": SemanticRetriever(index, embedder),
        "keyword": KeywordRetriever(index),
    }
    try:
        llm = OpenAIChatModel(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            timeout=settings.llm_timeout_seconds,
        )
    except LLMConfigurationError:
        llm = None
    return RAGPipeline(
        settings=settings,
        retrievers=retrievers,
        catalog=PatentCatalog.load(settings.patent_metadata_path),
        llm=llm,
        recorded=RecordedAnswers(settings.demo_cache_path),
    )
