# Evaluation

This project is evaluated at three levels: **retrieval** (did we find the right evidence?), **answers** (is the generated text supported by that evidence?), and **system behavior** (latency, refusals, failure handling). Everything below reports only numbers that were actually measured; anything still to be run is marked **PENDING**.

## 1. Answer quality: original manual evaluation (v1.0, December 2025)

Source: `notebooks/04_rag_qa.ipynb` (cells 22–30), summarized in `MODEL_CARD.md`.

- 8 hand-written questions, `top_k=5`, `gpt-4.1-mini`, the **original v1.0 prompt** (before the v1.1 grounding/citation prompt).
- Each answer was manually labeled for retrieval success, support level and hallucination severity.

| Metric | Result |
|---|---|
| Retrieval success (≥1 relevant chunk in top 5) | 8 / 8 |
| Relevant chunk at the top position | 8 / 8 |
| Support: fully / mostly / moderately / partially / weakly | 1 / 4 / 1 / 1 / 1 |
| Hallucination: none / mild / moderate / severe | 1 / 6 / 1 / 0 |
| Mean hallucination severity (0 = none … 3 = severe) | 1.0 |

Caveats: one annotator, 8 questions, broad questions with many acceptable answers, and "relevant" judged by inspection without a predefined answer key. (The original README listed "mostly supported" as 5 / 62.5%. The notebook data shows 4 / 50%; this has been corrected in `MODEL_CARD.md`.)

## 2. Retrieval quality: known-item evaluation (v1.1)

**Script:** `python scripts/evaluate.py` · **Data:** `eval/retrieval_eval_set.json` · **Outputs:** `eval/results/`

### Method

- **39 in-scope queries**, each labeled with the patent(s) it targets:
  - `abstract_paraphrase` (31): one per patent family, paraphrasing that patent's abstract or title while mostly avoiding its exact title wording.
  - `lay_language` (8): everyday wording that deliberately differs from the patent's vocabulary (for example, "a chatbot that answers questions about pictures" for a *visual dialogue* patent). These test meaning-based matching.
- **5 out-of-scope queries** with no relevant patent (for example, CRISPR or sourdough), used to calibrate the relevance threshold.
- **Metrics (patent level):** Hit@k = fraction of queries with a chunk from a relevant patent in the top k (equals Recall@k when one patent is relevant). MRR@10 = mean of 1/rank of the first relevant chunk. Retrieval latency excludes one-time model loading.
- **Label provenance:** single annotator. The labels were drafted with AI assistance while preparing v1.1 and should be reviewed by the project owner. Known-item queries measure whether the target document is found, not whether every relevant passage is found.

### Results

**Keyword (TF-IDF) baseline: measured 2026-10-14** (`eval/results/retrieval_keyword_20261014_0431.md`)

| Query subset | n | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|
| all | 39 | 0.82 | 0.85 | 0.87 | 0.85 |
| abstract_paraphrase | 31 | 0.94 | 0.97 | 1.00 | 0.96 |
| lay_language | 8 | 0.38 | 0.38 | 0.38 | 0.42 |

Keyword retrieval latency: 1.5 ms mean, 1.6 ms p95 (2,905 chunks, single CPU process).

Score separation for keyword search: in-scope top scores ranged from 0.100 to 0.478 (median 0.278), while out-of-scope queries reached up to 0.210. TF-IDF scores therefore **cannot reliably separate answerable from unanswerable questions**, which is why keyword mode relies on the LLM's insufficient-evidence check as a second safeguard.

**Semantic (all-MiniLM-L6-v2): PENDING.** The embedding model could not be downloaded in the environment where v1.1 was built. Run:

```bash
python scripts/evaluate.py
```

Then copy the table from the newest `eval/results/retrieval_semantic_keyword_*.md` here.

**What to look for:** keyword search is strong when queries reuse the patent's own words (abstract paraphrases) and weak on vocabulary mismatch (lay language: 3/8). The hypothesis that semantic search improves the `lay_language` subset is **not yet verified**. If it does not, say so; that is still a useful finding.

### Relevance threshold calibration

The app declines to answer when the best semantic match is below `MIN_SIMILARITY` (default 0.30). For reference, notebook runs in December 2025 produced top similarities of 0.52–0.58 for in-scope questions, so 0.30 sits comfortably below typical in-scope scores. Out-of-scope scores for MiniLM have **not yet been measured**. `scripts/evaluate.py` reports the in-scope minimum and out-of-scope maximum; choose a threshold between them and set `MIN_SIMILARITY` in `.env`. `scripts/check_setup.py` also reports whether the CRISPR question would be declined.

## 3. Answer quality with the v1.1 prompt: manual procedure (PENDING)

The v1.1 prompt adds numbered sources, mandatory `[S#]` citations, an `INSUFFICIENT_EVIDENCE` response, and a "from the patents" vs. "synthesis" structure. It has **not yet been evaluated against the live model**. To evaluate it:

1. `python scripts/evaluate.py --generate` runs the demo questions end to end and saves answers, citations and latencies to `eval/results/*.json`, each with an empty `manual_review` field.
2. For each answer, split it into individual claims. For each claim, open the cited source in the app and mark it:
   - **Supported**: the cited passage states it.
   - **Partially supported**: the passage implies it, or only part of it.
   - **Unsupported**: not in the cited passage (hallucination or wrong citation).
3. Report: % of claims supported, % of claims with any citation, the invalid-citation count (computed automatically), and whether the out-of-scope question was declined.
4. Compare with the v1.0 results in section 1 using the same 8 questions from notebook 04 to check whether the new prompt reduced unsupported claims.

### Automated checks already in place (unit-tested)

- Citations referring to sources that were never retrieved are flagged (`patent_rag/citations.py`).
- Answers without any citation are flagged as possibly ungrounded.
- Questions below the relevance threshold never reach the LLM (`tests/test_pipeline.py`).

## 4. System behavior

| Check | How it is verified | Status |
|---|---|---|
| Chunking reproduces the committed index exactly | `scripts/build_index.py --check`, `tests/test_ingest.py` | Pass |
| Numpy cosine search = notebooks' `NearestNeighbors` ranking | `tests/test_retrieval_and_index.py` | Pass |
| Missing key, outage, rate limit, malformed response → retrieval-only | `tests/test_pipeline.py`, `tests/test_llm.py` | Pass |
| Prompt-injection delimiter escape blocked | `tests/test_prompts.py` | Pass |
| UI renders answer, sources, fallbacks without exceptions | Browser check with stand-in models | Pass (UI only) |
| End-to-end latency with real models | `scripts/check_setup.py --ping`, app metrics | PENDING |

## 5. How to extend this evaluation

1. **Grow the labeled set** to around 100 queries written by someone other than the system builder, including multi-patent questions with graded relevance (0/1/2) so nDCG@k can be computed.
2. **Pool relevance judgments**: label the union of top-10 results from both retrievers so recall is not biased toward one method.
3. **Compare configurations** with the same script: chunk size (150/300/500 words), overlap, top-k, embedding models (for example `all-mpnet-base-v2` or `bge-small-en-v1.5`), and hybrid keyword + semantic ranking.
4. **Automate answer grading** with an LLM-as-judge for faithfulness, calibrated against the manual labels before trusting it.
5. **Track over time** in CI so changes to prompts, chunking or models cannot silently degrade quality.
