"""Ingestion: PDF text extraction, cleaning and chunking.

These functions are the same logic as notebooks 01_extract_text and 02_chunk, moved
into a module so they can be tested and reused by scripts/build_index.py.
Note: several patents in this corpus had text pasted in manually (OCR) because PDF
extraction failed; the existing .txt files in data/processed/txt are the source of truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text with pdfminer, falling back to pypdf. Returns "" if both fail."""
    text = ""
    try:
        import pdfminer.high_level

        text = pdfminer.high_level.extract_text(str(pdf_path)) or ""
    except Exception:
        text = ""
    if not text.strip():
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""
    return text


def clean_text(text: str) -> str:
    """Remove page numbers and figure labels, join lines, collapse whitespace (notebook 01)."""
    text = re.sub(r"^\s*\d+\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"FIG\.?\s*\d+.*", " ", text)
    text = re.sub(r"Fig\.?\s*\d+.*", " ", text)
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> list[str]:
    """Split text into overlapping word windows (notebook 02). Step = size - overlap."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_size_words")
    words = text.split()
    chunks: list[str] = []
    start, n = 0, len(words)
    while start < n:
        end = start + chunk_size_words
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap_words
    return chunks


def load_patent_texts(txt_dir: Path) -> dict[str, str]:
    """Load data/processed/txt/*.txt keyed by patent ID (file stem). Skips empty files."""
    docs = {}
    for txt_file in sorted(txt_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs[txt_file.stem] = text
    return docs


def build_chunk_records(
    docs: dict[str, str], chunk_size_words: int = 300, overlap_words: int = 50
) -> list[dict]:
    records = []
    for patent_id, text in docs.items():
        for i, chunk in enumerate(chunk_text(text, chunk_size_words, overlap_words)):
            records.append({"id": f"{patent_id}_{i}", "patent_id": patent_id, "chunk_index": i, "text": chunk})
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
