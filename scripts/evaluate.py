"""Retrieval evaluation: semantic vs keyword search on a small labeled query set.

Usage
    python scripts/evaluate.py                      # semantic + keyword (needs embedding model)
    python scripts/evaluate.py --methods keyword    # keyword baseline only (no model needed)
    python scripts/evaluate.py --generate           # also run end-to-end answers for the demo
                                                    # questions (needs OPENAI_API_KEY; costs a few cents)

Metrics (patent level; a hit = any retrieved chunk from a relevant patent):
    Hit@1, Hit@3, Hit@5  (= Recall@k when a query has one relevant patent)
    MRR                  (1 / rank of the first chunk from a relevant patent; 0 if not in top 10)
    Retrieval latency    (mean / p50 / p95, ms, excluding model load)
    Out-of-scope         (top similarity for questions the corpus cannot answer -> threshold calibration)

Results are printed and written to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patent_rag.config import get_settings  # noqa: E402
from patent_rag.index import load_index  # noqa: E402
from patent_rag.retrieval import KeywordRetriever, SemanticRetriever  # noqa: E402

EVAL_SET = ROOT / "eval" / "retrieval_eval_set.json"
RESULTS_DIR = ROOT / "eval" / "results"
DEPTH = 10
DEMO_QUESTIONS = [
    "How can a large language model be used to train or improve a smaller language model?",
    "How do the patents use large language models to generate training data for dialogue systems?",
    "How does the attention-free transformer reduce computational cost compared with standard attention?",
    "What does this collection say about CRISPR gene editing in crops?",
]


def first_relevant_rank(chunks, relevant: set[str]) -> int | None:
    for c in chunks:
        if c.patent_id in relevant:
            return c.rank
    return None


def percentile(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(rows) -> dict:
    n = len(rows)

    def hit(k):
        return sum(1 for r in rows if r["first_relevant_rank"] and r["first_relevant_rank"] <= k) / n

    return {
        "n_queries": n,
        "hit@1": round(hit(1), 3),
        "hit@3": round(hit(3), 3),
        "hit@5": round(hit(5), 3),
        "mrr@10": round(sum(1 / r["first_relevant_rank"] for r in rows if r["first_relevant_rank"]) / n, 3),
    }


def evaluate_method(retriever, eval_set) -> dict:
    retriever.search("warm-up query", DEPTH)  # exclude one-time warm-up from latency
    rows, latencies = [], []
    for item in eval_set["in_scope"]:
        res = retriever.search(item["query"], DEPTH)
        latencies.append(res.latency_ms)
        rows.append(
            {
                "query": item["query"],
                "type": item.get("type", "other"),
                "relevant": item["relevant"],
                "first_relevant_rank": first_relevant_rank(res.chunks, set(item["relevant"])),
                "top3_patents": [c.patent_id for c in res.chunks[:3]],
                "top_score": round(res.top_score, 4),
            }
        )
    by_type = {t: summarize([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
    oos = []
    for q in eval_set["out_of_scope"]:
        res = retriever.search(q, 5)
        oos.append({"query": q, "top_score": round(res.top_score, 4), "top_patent": res.chunks[0].patent_id})
    return {
        "overall": summarize(rows),
        "by_type": by_type,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
        },
        "in_scope_top_score": {
            "min": round(min(r["top_score"] for r in rows), 4),
            "median": round(statistics.median(r["top_score"] for r in rows), 4),
        },
        "out_of_scope": oos,
        "per_query": rows,
    }


def run_generation(settings) -> list[dict]:
    from patent_rag.factory import build_pipeline

    pipeline = build_pipeline(settings)
    if pipeline.llm is None:
        print("OPENAI_API_KEY not set; skipping --generate.")
        return []
    out = []
    for q in DEMO_QUESTIONS:
        resp = pipeline.answer(q)
        out.append(
            {
                "question": q,
                "status": resp.status,
                "evidence_patents": [c.patent_id for c in resp.evidence],
                "citations_valid": resp.citations.cited,
                "citations_invalid": resp.citations.invalid,
                "retrieval_ms": round(resp.timings_ms.get("retrieval", 0), 1),
                "generation_ms": round(resp.timings_ms.get("generation", 0), 1),
                "total_ms": round(resp.timings_ms.get("total", 0), 1),
                "answer": resp.answer,
                "manual_review": {"all_claims_supported": None, "notes": ""},
            }
        )
        print(f"- [{resp.status}] {q}  ({resp.timings_ms.get('total', 0) / 1000:.1f}s, cites {resp.citations.cited})")
    return out


def to_markdown(results: dict, settings) -> str:
    lines = [
        f"# Retrieval evaluation results ({results['run_at']})",
        "",
        f"Eval set: `eval/retrieval_eval_set.json` ({results['n_in_scope']} in-scope known-item queries, "
        f"{results['n_out_of_scope']} out-of-scope). Index: {results['index_size']} chunks. "
        f"Embedding model: `{settings.embedding_model}`.",
        "",
        "| Method | Query subset | n | Hit@1 | Hit@3 | Hit@5 | MRR@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results["methods"].items():
        for subset, mm in [("all", m["overall"]), *m["by_type"].items()]:
            lines.append(
                f"| {name} | {subset} | {mm['n_queries']} | {mm['hit@1']:.2f} | {mm['hit@3']:.2f} | "
                f"{mm['hit@5']:.2f} | {mm['mrr@10']:.2f} |"
            )
    lines += ["", "| Method | Retrieval latency mean / p50 / p95 (ms) |", "|---|---|"]
    for name, m in results["methods"].items():
        lat = m["latency_ms"]
        lines.append(f"| {name} | {lat['mean']:.1f} / {lat['p50']:.1f} / {lat['p95']:.1f} |")
    lines += ["", "## Top score: in-scope vs out-of-scope (threshold calibration)", ""]
    lines += ["| Method | In-scope min | In-scope median | Out-of-scope max |", "|---|---|---|---|"]
    for name, m in results["methods"].items():
        oos_max = max(o["top_score"] for o in m["out_of_scope"])
        lines.append(
            f"| {name} | {m['in_scope_top_score']['min']:.3f} | {m['in_scope_top_score']['median']:.3f} | {oos_max:.3f} |"
        )
    for name, m in results["methods"].items():
        misses = [r for r in m["per_query"] if not r["first_relevant_rank"] or r["first_relevant_rank"] > 1]
        lines += ["", f"## {name}: queries where the target patent was not ranked first", ""]
        if not misses:
            lines.append("None.")
        for r in misses:
            lines.append(
                f"- rank {r['first_relevant_rank'] or '>10'} [{r['type']}] — {r['query']} "
                f"(expected {', '.join(r['relevant'])}; got {', '.join(r['top3_patents'])})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["semantic", "keyword"], choices=["semantic", "keyword"])
    parser.add_argument("--generate", action="store_true", help="also run end-to-end answers (uses the API)")
    args = parser.parse_args()

    settings = get_settings()
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    index = load_index(settings.embeddings_path, settings.chunk_metadata_path)

    retrievers = {}
    if "semantic" in args.methods:
        from patent_rag.embedder import SentenceTransformerEmbedder

        t0 = time.perf_counter()
        retrievers["semantic"] = SemanticRetriever(index, SentenceTransformerEmbedder(settings.embedding_model))
        print(f"Embedding model loaded in {time.perf_counter() - t0:.1f}s")
    if "keyword" in args.methods:
        retrievers["keyword"] = KeywordRetriever(index)

    results = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "index_size": index.size,
        "n_in_scope": len(eval_set["in_scope"]),
        "n_out_of_scope": len(eval_set["out_of_scope"]),
        "methods": {name: evaluate_method(r, eval_set) for name, r in retrievers.items()},
    }
    if args.generate:
        results["generation"] = run_generation(settings)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = "_".join(args.methods)
    (RESULTS_DIR / f"retrieval_{tag}_{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = to_markdown(results, settings)
    (RESULTS_DIR / f"retrieval_{tag}_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
