# Retrieval evaluation results (2026-10-14 04:31)

Eval set: `eval/retrieval_eval_set.json` (39 in-scope known-item queries, 5 out-of-scope). Index: 2905 chunks. Embedding model: `all-MiniLM-L6-v2`.

| Method | Query subset | n | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|---|
| keyword | all | 39 | 0.82 | 0.85 | 0.87 | 0.85 |
| keyword | abstract_paraphrase | 31 | 0.94 | 0.97 | 1.00 | 0.96 |
| keyword | lay_language | 8 | 0.38 | 0.38 | 0.38 | 0.42 |

| Method | Retrieval latency mean / p50 / p95 (ms) |
|---|---|
| keyword | 1.5 / 1.5 / 1.6 |

## Top score: in-scope vs out-of-scope (threshold calibration)

| Method | In-scope min | In-scope median | Out-of-scope max |
|---|---|---|---|
| keyword | 0.100 | 0.278 | 0.210 |

## keyword: queries where the target patent was not ranked first

- rank 5 [abstract_paraphrase] — A neural network that converts an input sequence to an output sequence using attention layers instead of recurrence (expected US10452978; got US11238332, US11238332, US11238332)
- rank 2 [abstract_paraphrase] — Extracting information from images of a device's user interface display captured by a camera (expected US11295552; got US12282696B2, US11295552, US12282696B2)
- rank 10 [lay_language] — How can a chatbot answer questions about pictures? (expected US11562147; got US11989527, US20230252224A1, US20230252224A1)
- rank 9 [lay_language] — How can a computer learn the meaning of words from large amounts of text? (expected US9037464, US9740680; got US11989527, US11989527, US11989527)
- rank 6 [lay_language] — A small AI model learning from a bigger one (expected US20240346254A1; got US20240185001A1, US20240185001A1, US20240185001A1)
- rank >10 [lay_language] — Telling whether a photo looks good or bad (expected US12217382; got US11989527, US20230252224A1, US20230252224A1)
- rank >10 [lay_language] — Keeping electric car batteries from overheating (expected US11749857; got US11295552, US12282696B2, US12282696B2)
