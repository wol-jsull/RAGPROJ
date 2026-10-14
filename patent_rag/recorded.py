"""Recorded answers: a clearly labeled backup for live demos.

Every successful generated answer is saved locally (demo_cache/, git-ignored) together
with the time it was generated. If the LLM is unavailable during a demo and the same
question was answered before, the app can show that RECORDED answer, labeled with its
timestamp, next to freshly retrieved passages. It never presents a recording as live.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _key(question: str, method: str, top_k: int) -> str:
    q = re.sub(r"\s+", " ", question.strip().lower())
    return f"{method}|{top_k}|{q}"


class RecordedAnswers:
    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, question: str, method: str, top_k: int) -> dict | None:
        return self._load().get(_key(question, method, top_k))

    def save(self, question: str, method: str, top_k: int, answer: str, model: str) -> None:
        data = self._load()
        data[_key(question, method, top_k)] = {
            "question": question,
            "answer": answer,
            "model": model,
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            log.warning("could not write recorded answers file")  # non-fatal
