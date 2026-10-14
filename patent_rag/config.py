"""Central configuration, read from environment variables (optionally a local .env file).

Nothing secret is ever hardcoded here: the OpenAI key is only read from the
environment and is never logged or displayed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win.

    Kept dependency-free on purpose; python-dotenv would also work.
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():  # -sig: tolerate Notepad BOM
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Data artifacts produced by notebooks 02 and 03
    embeddings_path: Path = PROJECT_ROOT / "embeddings" / "embeddings.npy"
    chunk_metadata_path: Path = PROJECT_ROOT / "embeddings" / "chunk_metadata.jsonl"
    patent_metadata_path: Path = PROJECT_ROOT / "data" / "patent_metadata.json"
    demo_cache_path: Path = PROJECT_ROOT / "demo_cache" / "recorded_answers.json"

    # Models (same choices as the original notebooks)
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    llm_timeout_seconds: float = 30.0

    # Retrieval
    top_k: int = 5
    max_top_k: int = 10
    # Chunks with cosine similarity below this are treated as "not relevant enough"
    # to send to the LLM. Calibrate with scripts/evaluate.py (see EVALUATION.md).
    min_similarity: float = 0.30

    # Input limits
    max_query_chars: int = 500
    min_query_chars: int = 3

    openai_api_key: str | None = field(default=None, repr=False)

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Build Settings from the environment (after loading .env if present)."""
    load_dotenv()
    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 30.0),
        top_k=_env_int("TOP_K", 5),
        min_similarity=_env_float("MIN_SIMILARITY", 0.30),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
    )
