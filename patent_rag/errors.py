"""Exception hierarchy with user-safe messages.

Every error raised to the UI carries a ``user_message`` that is safe to display:
no stack traces, no API keys, no raw provider payloads.
"""

from __future__ import annotations


class PatentRAGError(Exception):
    """Base class. ``user_message`` is safe to show in the interface."""

    user_message = "Something went wrong while answering your question."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None):
        if user_message:
            self.user_message = user_message
        # ``detail`` is for developer logs only and must never contain secrets.
        self.detail = detail
        super().__init__(detail or self.user_message)


class InvalidQueryError(PatentRAGError):
    user_message = "Please enter a valid question."


class IndexNotFoundError(PatentRAGError):
    user_message = (
        "The patent index files are missing. Run the notebooks in notebooks/ "
        "(or scripts/build_index.py) to create embeddings/embeddings.npy and "
        "embeddings/chunk_metadata.jsonl."
    )


class IndexIntegrityError(PatentRAGError):
    user_message = "The patent index files are inconsistent or corrupted. Rebuild the index."


class EmbeddingModelError(PatentRAGError):
    user_message = (
        "The embedding model could not be loaded. Check that sentence-transformers is "
        "installed and that the model can be downloaded (or is already cached)."
    )


class LLMConfigurationError(PatentRAGError):
    user_message = (
        "No OpenAI API key is configured. Set OPENAI_API_KEY to enable generated answers. "
        "Retrieved patent passages are still shown below."
    )


class LLMUnavailableError(PatentRAGError):
    user_message = (
        "The language model service is unavailable right now. "
        "Retrieved patent passages are still shown below."
    )


class LLMRateLimitError(LLMUnavailableError):
    user_message = (
        "The language model service is rate-limiting requests. Wait a moment and try again. "
        "Retrieved patent passages are still shown below."
    )


class LLMResponseError(PatentRAGError):
    user_message = (
        "The language model returned an empty or malformed response. "
        "Retrieved patent passages are still shown below."
    )
