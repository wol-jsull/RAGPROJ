"""Checking that an answer's inline citations point at sources that were actually retrieved."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[\s*S(\d+)\s*\]", re.IGNORECASE)


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)  # valid 1-based source positions, in order
    invalid: list[int] = field(default_factory=list)  # cited labels with no matching source

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)


def extract_citations(answer: str, num_sources: int) -> CitationReport:
    """Find [S#] labels; split them into valid (1..num_sources) and invalid ones."""
    report = CitationReport()
    for match in _CITATION_RE.finditer(answer or ""):
        n = int(match.group(1))
        bucket = report.cited if 1 <= n <= num_sources else report.invalid
        if n not in bucket:
            bucket.append(n)
    return report
