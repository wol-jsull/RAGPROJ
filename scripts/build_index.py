"""Rebuild the chunk file and vector index from data/processed/txt (scripted notebooks 02 + 03).

Usage
    python scripts/build_index.py --check        # re-chunk and compare to the committed index (no model needed)
    python scripts/build_index.py                # re-chunk + re-embed, writes embeddings/ (needs the model)

PDF -> text extraction (notebook 01) is not re-run: several patents needed manual OCR text,
so the .txt files in data/processed/txt are the source of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patent_rag.config import get_settings  # noqa: E402
from patent_rag.ingest import build_chunk_records, load_patent_texts, write_jsonl  # noqa: E402

TXT_DIR = ROOT / "data" / "processed" / "txt"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks" / "patent_chunks.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only verify chunking reproduces the committed index")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    settings = get_settings()

    docs = load_patent_texts(TXT_DIR)
    records = build_chunk_records(docs, args.chunk_size, args.overlap)
    print(f"{len(docs)} documents -> {len(records)} chunks (size={args.chunk_size}, overlap={args.overlap})")

    if args.check:
        with settings.chunk_metadata_path.open(encoding="utf-8") as f:
            committed = [json.loads(line) for line in f]
        same = committed == records
        print("Matches committed index." if same else "DIFFERS from committed index.")
        return 0 if same else 1

    from patent_rag.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    texts = [r["text"] for r in records]
    vectors = np.vstack([embedder.encode(texts[i : i + 64]) for i in range(0, len(texts), 64)])
    write_jsonl(records, CHUNKS_PATH)
    write_jsonl(records, settings.chunk_metadata_path)
    np.save(settings.embeddings_path, vectors.astype(np.float32))
    print(f"Wrote {vectors.shape} embeddings to {settings.embeddings_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
