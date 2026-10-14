"""Pre-demo checklist: dependencies, index files, embedding model, API key, and demo queries.

Usage
    python scripts/check_setup.py           # everything except a live LLM call
    python scripts/check_setup.py --ping    # also send one tiny request to the LLM (needs OPENAI_API_KEY)

Exit code 0 = ready for the demo (warnings allowed), 1 = something essential is broken.
The API key is never printed.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
problems = 0

# Demo questions and the patent we expect near the top (verify these tonight).
DEMO_EXPECTATIONS = [
    ("How can a large language model be used to train or improve a smaller language model?", "US20240346254A1"),
    ("How do the patents use large language models to generate training data for dialogue systems?", "US20240185001A1"),
    ("How does the attention-free transformer reduce computational cost compared with standard attention?", "US12271791B2"),
]
OUT_OF_SCOPE = "What does this collection say about CRISPR gene editing in crops?"


def report(status: str, msg: str) -> None:
    global problems
    if status == FAIL:
        problems += 1
    print(f"{status} {msg}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="make one small LLM request to verify the key/model")
    args = parser.parse_args()

    print("== Python and packages")
    v = sys.version_info
    report(OK if v >= (3, 10) else FAIL, f"Python {v.major}.{v.minor}.{v.micro} (3.10+ required)")
    for pkg in ["numpy", "sklearn", "openai", "streamlit", "sentence_transformers"]:
        try:
            mod = importlib.import_module(pkg)
            report(OK, f"{pkg} {getattr(mod, '__version__', '')}")
        except ImportError:
            report(FAIL, f"{pkg} not installed -> pip install -r requirements.txt")

    from patent_rag.config import get_settings
    from patent_rag.errors import PatentRAGError
    from patent_rag.index import load_index
    from patent_rag.metadata import PatentCatalog

    settings = get_settings()

    print("\n== Index files")
    try:
        index = load_index(settings.embeddings_path, settings.chunk_metadata_path)
        report(OK, f"{index.size} chunks, {index.dimension}-dim embeddings, {len(index.patent_ids)} patents")
    except PatentRAGError as exc:
        report(FAIL, f"{exc.user_message} ({exc.detail})")
        return 1
    catalog = PatentCatalog.load(settings.patent_metadata_path)
    missing = [p for p in index.patent_ids if not catalog.get(p).title]
    report(OK if not missing else WARN, f"titles available for {len(index.patent_ids) - len(missing)}/{len(index.patent_ids)} patents")

    print("\n== OpenAI configuration")
    if settings.has_api_key:
        report(OK, f"OPENAI_API_KEY is set (value hidden); model = {settings.llm_model}")
    else:
        report(WARN, "OPENAI_API_KEY not set -> app runs in retrieval-only mode (see .env.example)")

    print("\n== Embedding model (first run downloads ~90 MB)")
    try:
        from patent_rag.embedder import SentenceTransformerEmbedder
        from patent_rag.retrieval import SemanticRetriever

        t0 = time.perf_counter()
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        report(OK, f"{settings.embedding_model} loaded in {time.perf_counter() - t0:.1f}s")
        if embedder.dimension != index.dimension:
            report(FAIL, f"model dimension {embedder.dimension} != index dimension {index.dimension}")
            return 1
        retriever = SemanticRetriever(index, embedder)
    except PatentRAGError as exc:
        report(FAIL, f"{exc.user_message} ({exc.detail})")
        return 1

    print("\n== Demo questions (semantic retrieval, top 5)")
    for question, expected in DEMO_EXPECTATIONS:
        res = retriever.search(question, 5)
        patents = [c.patent_id for c in res.chunks]
        found = expected in patents
        report(
            OK if found else WARN,
            f"{question[:70]}...\n        top score {res.top_score:.3f}; expected {expected} "
            f"{'at rank ' + str(patents.index(expected) + 1) if found else 'NOT in top 5'}; "
            f"{res.latency_ms:.0f} ms",
        )
        if res.top_score < settings.min_similarity:
            report(WARN, f"top score below MIN_SIMILARITY={settings.min_similarity}; lower it or rephrase")
    res = retriever.search(OUT_OF_SCOPE, 5)
    declines = res.top_score < settings.min_similarity
    report(
        OK if declines else WARN,
        f"out-of-scope question top score {res.top_score:.3f} -> "
        f"{'declines (good)' if declines else 'would be sent to the LLM; consider raising MIN_SIMILARITY'}",
    )

    if args.ping:
        print("\n== LLM ping")
        if not settings.has_api_key:
            report(WARN, "skipped (no key)")
        else:
            from patent_rag.llm import OpenAIChatModel

            try:
                t0 = time.perf_counter()
                reply = OpenAIChatModel(settings.openai_api_key, settings.llm_model, timeout=20).complete(
                    [{"role": "user", "content": "Reply with the single word: ready"}]
                )
                report(OK, f"{settings.llm_model} replied '{reply[:20]}' in {time.perf_counter() - t0:.1f}s")
            except PatentRAGError as exc:
                report(FAIL, exc.user_message)

    print("\n" + ("READY for the demo." if problems == 0 else f"{problems} problem(s) found — fix the [FAIL] lines above."))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
