import json

import pytest

from patent_rag.ingest import build_chunk_records, chunk_text, clean_text, load_patent_texts

from conftest import PROJECT_ROOT


def test_chunk_windows_overlap_by_configured_amount():
    words = [f"w{i}" for i in range(10)]
    chunks = chunk_text(" ".join(words), chunk_size_words=4, overlap_words=1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_short_and_empty_texts():
    assert chunk_text("") == []
    assert chunk_text("just three words") == ["just three words"]


@pytest.mark.parametrize("size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size, overlap)


def test_clean_text_removes_page_numbers_figures_and_newlines():
    raw = "12 The encoder\nprocesses input.\nFIG. 3 shows a diagram\nThe decoder-\nworks."
    assert clean_text(raw) == "The encoder processes input. The decoderworks."


def test_rebuilding_chunks_reproduces_committed_index_exactly():
    """The packaged chunker is a faithful port of notebook 02 (300 words, 50 overlap)."""
    docs = load_patent_texts(PROJECT_ROOT / "data" / "processed" / "txt")
    rebuilt = build_chunk_records(docs, 300, 50)
    with (PROJECT_ROOT / "embeddings" / "chunk_metadata.jsonl").open(encoding="utf-8") as f:
        committed = [json.loads(line) for line in f]
    assert rebuilt == committed
