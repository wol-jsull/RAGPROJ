"""Per-patent display metadata (title, number, applicant, link)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatentInfo:
    patent_id: str
    number: str
    title: str | None
    applicant: str | None
    doc_type: str | None
    url: str


def google_patents_url(patent_id: str) -> str:
    """Build a Google Patents link from the corpus identifier.

    Corpus IDs look like ``US10452978``, ``US12271791B2`` or ``US_12380282_B2``.
    The link is constructed (not stored in the source data), so the UI labels it as such.
    """
    compact = re.sub(r"[^A-Za-z0-9]", "", patent_id).upper()
    return f"https://patents.google.com/patent/{compact}/en"


class PatentCatalog:
    def __init__(self, records: dict[str, dict]):
        self._records = records

    @classmethod
    def load(cls, path: Path) -> "PatentCatalog":
        """Load metadata; a missing/corrupt file degrades to IDs only (never crashes the app)."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records = data.get("patents", {})
            if not isinstance(records, dict):
                records = {}
        except (OSError, json.JSONDecodeError):
            records = {}
        return cls(records)

    def get(self, patent_id: str) -> PatentInfo:
        rec = self._records.get(patent_id, {})
        return PatentInfo(
            patent_id=patent_id,
            number=rec.get("number") or patent_id,
            title=rec.get("title"),
            applicant=rec.get("applicant"),
            doc_type=rec.get("type"),
            url=google_patents_url(patent_id),
        )

    def __len__(self) -> int:
        return len(self._records)
