"""Add patents and rebuild the index: extract new PDFs, chunk, embed (scripted notebooks 01-03).

Usage
    python scripts/build_index.py --check        # re-chunk and compare to the committed index (no model needed)
    python scripts/build_index.py                # extract NEW PDFs, re-chunk, re-embed everything

Adding patents
    1. Put the new PDFs in data/raw/patents/ (file name = patent ID, e.g. US11222333B2.pdf)
    2. python scripts/build_index.py
    3. Review the auto-added entries in data/patent_metadata.json (marked "auto": true)

Existing .txt files in data/processed/txt are never overwritten (several were fixed by hand
with OCR). PDFs that yield little or no text (scanned images) are skipped with a warning:
OCR them and save the text as data/processed/txt/<ID>.txt, then run this script again.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patent_rag.config import get_settings  # noqa: E402
from patent_rag.ingest import (  # noqa: E402
    build_chunk_records,
    clean_text,
    load_patent_texts,
    pdf_to_text,
    write_jsonl,
)

RAW_DIR = ROOT / "data" / "raw" / "patents"
TXT_DIR = ROOT / "data" / "processed" / "txt"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks" / "patent_chunks.jsonl"
MIN_WORDS = 200  # less than this from a whole patent almost always means a scanned PDF


def extract_new_pdfs() -> list[str]:
    """Extract text for PDFs that don't have a .txt yet. Returns the IDs that were added."""
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    added = []
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        out = TXT_DIR / f"{pdf.stem}.txt"
        if out.exists():
            continue
        text = clean_text(pdf_to_text(pdf))
        n_words = len(text.split())
        if n_words < MIN_WORDS:
            print(f"  WARNING {pdf.name}: only {n_words} words extracted (scanned PDF?). Skipped. "
                  f"OCR it and save the text as data/processed/txt/{out.name}")
            continue
        out.write_text(text, encoding="utf-8")
        added.append(pdf.stem)
        print(f"  extracted {pdf.name}: {n_words:,} words")
    return added


def guess_title(text: str) -> str | None:
    """Best-effort title from the front page's (54) field; OCR text is messy, so it's flagged for review."""
    m = re.search(r"\(\s*54\s*\)\s*(.{5,250}?)\s*\(\s*(?:7[1-5]|2[12]|5[1268]|60|63|65)\s*\)", text)
    return m.group(1).strip(" .,;").title() if m else None


def add_metadata_stubs(docs: dict[str, str], metadata_path: Path) -> dict[str, str | None]:
    data = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {"patents": {}}
    patents = data.setdefault("patents", {})
    new = [pid for pid in docs if pid not in patents]
    for pid in new:
        patents[pid] = {
            "number": pid,
            "type": "Patent application publication" if re.search(r"A\d$", pid) else "Granted patent",
            "title": guess_title(docs[pid]),
            "applicant": None,
            "auto": True,
        }
    if new:
        metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {pid: patents[pid]["title"] for pid in new}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only verify chunking reproduces the committed index")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    settings = get_settings()

    if not args.check:
        print("Step 1/3: extracting text from new PDFs")
        added = extract_new_pdfs()
        print(f"  {len(added)} new patent(s) extracted" if added else "  no new PDFs")

    docs = load_patent_texts(TXT_DIR)
    records = build_chunk_records(docs, args.chunk_size, args.overlap)
    print(f"{len(docs)} documents -> {len(records)} chunks (size={args.chunk_size}, overlap={args.overlap})")

    if args.check:
        with settings.chunk_metadata_path.open(encoding="utf-8") as f:
            committed = [json.loads(line) for line in f]
        same = committed == records
        print("Matches committed index." if same else "DIFFERS from committed index.")
        return 0 if same else 1

    print("Step 2/3: embedding all chunks (a few minutes on a laptop CPU)")
    from patent_rag.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    texts = [r["text"] for r in records]
    batches = []
    for i in range(0, len(texts), 64):
        batches.append(embedder.encode(texts[i : i + 64]))
        print(f"  {min(i + 64, len(texts))}/{len(texts)}", end="\r")
    vectors = np.vstack(batches).astype(np.float32)
    write_jsonl(records, CHUNKS_PATH)
    write_jsonl(records, settings.chunk_metadata_path)
    np.save(settings.embeddings_path, vectors)
    print(f"\n  wrote {vectors.shape} embeddings to {settings.embeddings_path}")

    print("Step 3/3: patent metadata")
    new_meta = add_metadata_stubs(docs, settings.patent_metadata_path)
    for pid, title in new_meta.items():
        print(f"  added {pid}: title guess = {title!r}")
    if new_meta:
        print("  Review these in data/patent_metadata.json (fix titles, add applicant, delete \"auto\").")
    print("Done. Restart the Streamlit app to load the new index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
