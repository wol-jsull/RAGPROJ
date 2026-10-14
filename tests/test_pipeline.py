import pytest

from patent_rag.errors import InvalidQueryError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from patent_rag.pipeline import ANSWERED, INSUFFICIENT, RECORDED, RETRIEVAL_ONLY
from patent_rag.recorded import RecordedAnswers

from conftest import FakeLLM


def test_answered_with_valid_citations(make_pipeline):
    llm = FakeLLM()
    resp = make_pipeline(llm=llm).answer("How does sparse attention work?", top_k=3)
    assert resp.status == ANSWERED
    assert resp.citations.cited == [1]
    assert resp.notice is None
    assert resp.model == "fake-model"
    # Only chunks above the 0.3 threshold are sent: PA_0 (1.0) and PA_1 (~0.99); PB_0 scores 0.
    assert [c.chunk_id for c in resp.evidence] == ["PA_0", "PA_1"]
    assert len(resp.retrieval.chunks) == 3
    assert "Sparse attention mechanism" in llm.calls[0][1]["content"]
    assert set(resp.timings_ms) >= {"retrieval", "generation", "total"}


def test_low_similarity_skips_llm_and_reports_insufficient_evidence(make_pipeline):
    llm = FakeLLM()
    resp = make_pipeline(llm=llm).answer("What about CRISPR gene editing?", top_k=3)
    assert resp.status == INSUFFICIENT
    assert llm.calls == []  # the model is never asked to answer without evidence
    assert resp.evidence == []
    assert "declined" in resp.notice
    assert len(resp.retrieval.chunks) == 3  # closest passages still shown for transparency


def test_model_can_declare_insufficient_evidence(make_pipeline):
    llm = FakeLLM(reply="INSUFFICIENT_EVIDENCE: the sources do not discuss licensing terms.")
    resp = make_pipeline(llm=llm).answer("What licensing terms apply to attention?")
    assert resp.status == INSUFFICIENT
    assert resp.answer == "the sources do not discuss licensing terms."


def test_answer_without_citations_gets_warning(make_pipeline):
    resp = make_pipeline(llm=FakeLLM(reply="Attention is great.")).answer("attention?")
    assert resp.status == ANSWERED
    assert "no source citations" in resp.notice


def test_answer_with_out_of_range_citation_gets_warning(make_pipeline):
    resp = make_pipeline(llm=FakeLLM(reply="Fact [S1]. Made up [S7].")).answer("attention?")
    assert resp.citations.invalid == [7]
    assert "S7" in resp.notice


def test_missing_api_key_falls_back_to_retrieval_only(make_pipeline):
    resp = make_pipeline(llm=None).answer("attention?")
    assert resp.status == RETRIEVAL_ONLY
    assert "OPENAI_API_KEY" in resp.notice
    assert resp.evidence and resp.messages  # evidence + prompt still available to show


@pytest.mark.parametrize("error", [LLMUnavailableError(), LLMRateLimitError(), LLMResponseError()])
def test_llm_failures_degrade_to_retrieval_only(make_pipeline, error):
    resp = make_pipeline(llm=FakeLLM(error=error)).answer("attention?")
    assert resp.status == RETRIEVAL_ONLY
    assert resp.notice == error.user_message
    assert resp.answer is None and resp.evidence


def test_recorded_answer_is_used_and_labeled_when_llm_fails(make_pipeline, settings):
    recorded = RecordedAnswers(settings.demo_cache_path)
    make_pipeline(llm=FakeLLM(), recorded=recorded).answer("How does sparse attention work?", top_k=3)

    resp = make_pipeline(llm=FakeLLM(error=LLMUnavailableError()), recorded=recorded).answer(
        "  how does SPARSE attention work? ", top_k=3
    )
    assert resp.status == RECORDED
    assert "RECORDED" in resp.notice and resp.recorded_at
    assert resp.citations.cited == [1]


def test_recorded_answer_not_reused_for_different_settings(make_pipeline, settings):
    recorded = RecordedAnswers(settings.demo_cache_path)
    make_pipeline(llm=FakeLLM(), recorded=recorded).answer("attention?", top_k=3)
    resp = make_pipeline(llm=FakeLLM(error=LLMUnavailableError()), recorded=recorded).answer("attention?", top_k=2)
    assert resp.status == RETRIEVAL_ONLY


def test_generation_toggle_off_never_calls_llm(make_pipeline):
    llm = FakeLLM()
    resp = make_pipeline(llm=llm).answer("attention?", generate=False)
    assert resp.status == RETRIEVAL_ONLY and llm.calls == []


def test_invalid_query_raises_before_retrieval(make_pipeline):
    with pytest.raises(InvalidQueryError):
        make_pipeline(llm=FakeLLM()).answer("   ")


def test_top_k_is_clamped_to_configured_maximum(make_pipeline):
    resp = make_pipeline(llm=FakeLLM(), max_top_k=2).answer("attention?", top_k=99)
    assert len(resp.retrieval.chunks) == 2
