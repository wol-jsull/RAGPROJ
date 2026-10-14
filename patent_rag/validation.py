"""User input validation."""

from __future__ import annotations

import re

from .errors import InvalidQueryError

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_query(raw: str | None, *, min_chars: int = 3, max_chars: int = 500) -> str:
    """Return a cleaned question or raise InvalidQueryError with a user-facing message."""
    if raw is None:
        raise InvalidQueryError("Please enter a question about the patent collection.")
    query = _CONTROL_CHARS_RE.sub(" ", str(raw))
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        raise InvalidQueryError("Please enter a question about the patent collection.")
    if len(query) < min_chars:
        raise InvalidQueryError(f"Your question is too short (minimum {min_chars} characters).")
    if len(query) > max_chars:
        raise InvalidQueryError(
            f"Your question is too long ({len(query)} characters; maximum {max_chars}). "
            "Try asking one focused question."
        )
    if not re.search(r"[A-Za-z0-9]", query):
        raise InvalidQueryError("Your question must contain letters or numbers.")
    return query
