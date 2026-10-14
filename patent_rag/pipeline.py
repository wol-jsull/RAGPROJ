"""End-to-end RAG orchestration.

    question -> validate -> retrieve top-k -> relevance gate -> build prompt -> LLM -> check citations

Each stage degrades gracefully: if generation is impossible (no key, outage, rate limit,
malformed output) the user still gets the retrieved evidence plus a clear notice; if
retrieval finds nothing relevant enough, the LLM is never called.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

from .citations import CitationReport, extract_citations
from .config import Settings
from .errors import LLMConfigurationError, PatentRAGError
from .llm import ChatModel
from .metadata import PatentCatalog
from .prompts import INSUFFICIENT_EVIDENCE_TOKEN, build_messages
from .recorded import RecordedAnswers
from .retrieval import RetrievalResult, RetrievedChunk
from .validation import validate_query

log = logging.getLogger(__name__)

# Answer statuses shown in the UI
ANSWERED = "answered"  # live LLM answer grounded in retrieved sources
INSUFFICIENT = "insufficient_evidence"  # gate or model said the sources don't support an answer
RETRIEVAL_ONLY = "retrieval_only"  # no generation (disabled, no key, or LLM failure)
RECORDED = "recorded"  # LLM failed; showing a previously recorded answer, clearly labeled

# TF-IDF cosine scores live on a different scale from embedding cosine similarity.
# Not calibrated: off-topic questions can still score ~0.15 on generic words, so the
# keyword mode relies more heavily on the model's INSUFFICIENT_EVIDENCE check.
MIN_KEYWORD_SCORE = 0.10


@dataclass
class RAGResponse:
    question: str
    status: str
    retrieval: RetrievalResult
    evidence: list[RetrievedChunk]  # chunks actually passed to the LLM (labels S1..Sn)
    answer: str | None = None
    notice: str | None = None  # user-facing explanation (fallbacks, warnings)
    citations: CitationReport = field(default_factory=CitationReport)
    messages: list[dict[str, str]] | None = None  # exact prompt, for transparency
    model: str | None = None
    recorded_at: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retrievers: dict,
        catalog: PatentCatalog,
        llm: ChatModel | None = None,
        recorded: RecordedAnswers | None = None,
    ):
        self.settings = settings
        self.retrievers = retrievers  # {"semantic": SemanticRetriever, "keyword": KeywordRetriever}
        self.catalog = catalog
        self.llm = llm
        self.recorded = recorded

    def threshold(self, method: str) -> float:
        return self.settings.min_similarity if method == "semantic" else MIN_KEYWORD_SCORE

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        method: str = "semantic",
        patent_filter: Iterable[str] | None = None,
        generate: bool = True,
    ) -> RAGResponse:
        t0 = time.perf_counter()
        q = validate_query(
            question,
            min_chars=self.settings.min_query_chars,
            max_chars=self.settings.max_query_chars,
        )
        top_k = max(1, min(top_k or self.settings.top_k, self.settings.max_top_k))
        if method not in self.retrievers:
            raise PatentRAGError(f"Unknown retrieval method '{method}'.")

        retrieval = self.retrievers[method].search(q, top_k, patent_filter)
        threshold = self.threshold(method)
        evidence = [c for c in retrieval.chunks if c.score >= threshold]
        resp = RAGResponse(question=q, status=RETRIEVAL_ONLY, retrieval=retrieval, evidence=evidence)
        resp.timings_ms["retrieval"] = retrieval.latency_ms

        if not evidence:
            resp.status = INSUFFICIENT
            resp.notice = (
                f"No passage was relevant enough to answer from (best match scored "
                f"{retrieval.top_score:.2f}; threshold {threshold:.2f}). The assistant declined to "
                "generate an answer rather than guess. The closest passages are shown for reference."
                if retrieval.chunks
                else "No passages were retrieved for this question."
            )
            return self._finish(resp, t0)

        resp.messages = build_messages(q, evidence, self.catalog)
        if not generate:
            resp.notice = "Generation is turned off: showing retrieved evidence only."
            return self._finish(resp, t0)

        if self.llm is None:
            return self._fallback(resp, LLMConfigurationError(), method, top_k, t0)

        g0 = time.perf_counter()
        try:
            text = self.llm.complete(resp.messages)
        except PatentRAGError as exc:
            log.warning("generation failed: %s", exc.detail or type(exc).__name__)
            return self._fallback(resp, exc, method, top_k, t0)
        finally:
            resp.timings_ms["generation"] = (time.perf_counter() - g0) * 1000

        resp.model = self.llm.model
        stripped = text.lstrip()
        if stripped.upper().startswith(INSUFFICIENT_EVIDENCE_TOKEN):
            resp.status = INSUFFICIENT
            explanation = stripped[len(INSUFFICIENT_EVIDENCE_TOKEN):].strip(" :\n-")
            resp.answer = explanation or "The retrieved passages do not contain enough information."
            resp.notice = "The model judged that the retrieved passages do not support an answer."
            return self._finish(resp, t0)

        resp.status = ANSWERED
        resp.answer = text
        resp.citations = extract_citations(text, len(evidence))
        if not resp.citations.has_citations:
            resp.notice = (
                "This answer contains no source citations, so it may not be grounded in the "
                "retrieved passages. Verify it against the sources below."
            )
        elif resp.citations.invalid:
            bad = ", ".join(f"S{n}" for n in resp.citations.invalid)
            resp.notice = f"The answer cites source labels that do not exist ({bad}); treat those statements with caution."
        if self.recorded is not None:
            self.recorded.save(q, method, top_k, text, self.llm.model)
        return self._finish(resp, t0)

    def _fallback(self, resp: RAGResponse, exc: PatentRAGError, method: str, top_k: int, t0: float):
        record = self.recorded.get(resp.question, method, top_k) if self.recorded else None
        if record:
            resp.status = RECORDED
            resp.answer = record["answer"]
            resp.model = record.get("model")
            resp.recorded_at = record.get("recorded_at")
            resp.citations = extract_citations(resp.answer, len(resp.evidence))
            resp.notice = (
                f"{exc.user_message} Showing a RECORDED answer generated earlier "
                f"({resp.recorded_at}), not a live response."
            )
        else:
            resp.status = RETRIEVAL_ONLY
            resp.notice = exc.user_message
        return self._finish(resp, t0)

    @staticmethod
    def _finish(resp: RAGResponse, t0: float) -> RAGResponse:
        resp.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        log.info(
            "query len=%d status=%s method=%s retrieved=%d evidence=%d total_ms=%.0f",
            len(resp.question), resp.status, resp.retrieval.method,
            len(resp.retrieval.chunks), len(resp.evidence), resp.timings_ms["total"],
        )
        return resp
