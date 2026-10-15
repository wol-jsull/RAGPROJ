import json

import numpy as np
import pytest

from patent_rag.errors import IndexIntegrityError, IndexNotFoundError
from patent_rag.index import load_index
from patent_rag.metadata import PatentCatalog, google_patents_url
from patent_rag.retrieval import KeywordRetriever, SemanticRetriever

from conftest import PROJECT_ROOT, KeywordFakeEmbedder

EMB = PROJECT_ROOT / "embeddings" / "embeddings.npy"
META = PROJECT_ROOT / "embeddings" / "chunk_metadata.jsonl"


# ---------------------------------------------------------------- semantic retrieval (tiny index)
def test_semantic_ranking_and_scores(tiny_index):
    r = SemanticRetriever(tiny_index, KeywordFakeEmbedder()).search("attention please", top_k=3)
    assert [c.chunk_id for c in r.chunks] == ["PA_0", "PA_1", "PB_0"]
    assert [c.rank for c in r.chunks] == [1, 2, 3]
    assert r.chunks[0].score == pytest.approx(1.0)
    assert r.chunks[0].score >= r.chunks[1].score >= r.chunks[2].score
    assert r.method == "semantic" and r.latency_ms >= 0


def test_top_k_larger_than_index_is_clamped(tiny_index):
    r = SemanticRetriever(tiny_index, KeywordFakeEmbedder()).search("dialogue", top_k=50)
    assert len(r.chunks) == tiny_index.size


def test_patent_filter_restricts_results(tiny_index):
    r = SemanticRetriever(tiny_index, KeywordFakeEmbedder()).search("attention", top_k=5, patent_filter=["PC"])
    assert [c.patent_id for c in r.chunks] == ["PC"]


def test_embedding_dimension_mismatch_is_detected(tiny_index):
    class WrongDim:
        def encode(self, texts):
            return np.ones((1, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="dimension"):
        SemanticRetriever(tiny_index, WrongDim()).search("anything", top_k=1)


# ---------------------------------------------------------------- real index
def test_real_index_loads_and_is_consistent():
    index = load_index(EMB, META)
    txt_ids = {p.stem for p in (PROJECT_ROOT / "data" / "processed" / "txt").glob("*.txt")}
    assert index.size > 0
    assert index.dimension == 384
    assert set(index.patent_ids) == txt_ids  # every text file is indexed, nothing extra
    norms = np.linalg.norm(index.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_dot_product_search_matches_original_sklearn_nearest_neighbors():
    """The app's numpy search must rank exactly like notebook 04's NearestNeighbors(metric='cosine')."""
    from sklearn.neighbors import NearestNeighbors

    index = load_index(EMB, META)
    raw = np.load(EMB)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(raw)
    retriever = SemanticRetriever(index, embedder=None)
    rng = np.random.default_rng(0)
    for i in rng.choice(index.size, size=10, replace=False):
        query = raw[i] + rng.normal(0, 0.02, size=raw.shape[1]).astype(np.float32)
        query /= np.linalg.norm(query)
        dist, idx = nn.kneighbors(query[None, :])
        ours = retriever.search_vector(query, top_k=5)
        assert [index.chunks[j].id for j in idx[0]] == [c.chunk_id for c in ours]
        assert np.allclose(1 - dist[0], [c.score for c in ours], atol=1e-5)


def test_keyword_baseline_finds_exact_term_matches():
    index = load_index(EMB, META)
    r = KeywordRetriever(index).search("heat-radiating fluid battery module", top_k=3)
    assert r.chunks[0].patent_id == "US11749857"


# ---------------------------------------------------------------- index failure modes
def test_missing_index_files_raise_user_safe_error(tmp_path):
    with pytest.raises(IndexNotFoundError) as exc:
        load_index(tmp_path / "nope.npy", tmp_path / "nope.jsonl")
    assert "notebooks" in exc.value.user_message


def test_row_count_mismatch_is_detected(tmp_path):
    np.save(tmp_path / "e.npy", np.ones((3, 4), dtype=np.float32))
    (tmp_path / "m.jsonl").write_text(
        json.dumps({"id": "a_0", "patent_id": "a", "chunk_index": 0, "text": "t"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(IndexIntegrityError):
        load_index(tmp_path / "e.npy", tmp_path / "m.jsonl")


def test_malformed_metadata_is_detected(tmp_path):
    np.save(tmp_path / "e.npy", np.ones((1, 4), dtype=np.float32))
    (tmp_path / "m.jsonl").write_text('{"id": "a_0", "text": "missing fields"}\n', encoding="utf-8")
    with pytest.raises(IndexIntegrityError):
        load_index(tmp_path / "e.npy", tmp_path / "m.jsonl")


# ---------------------------------------------------------------- metadata / source attribution
def test_every_corpus_patent_has_display_metadata():
    index = load_index(EMB, META)
    catalog = PatentCatalog.load(PROJECT_ROOT / "data" / "patent_metadata.json")
    for pid in index.patent_ids:
        info = catalog.get(pid)
        assert info.title, pid
        assert info.number.startswith("US"), pid


def test_google_patents_url_normalizes_identifiers():
    assert google_patents_url("US_12380282_B2") == "https://patents.google.com/patent/US12380282B2/en"
    assert google_patents_url("US10452978") == "https://patents.google.com/patent/US10452978/en"


def test_missing_metadata_file_degrades_to_ids(tmp_path):
    info = PatentCatalog.load(tmp_path / "missing.json").get("US123")
    assert info.number == "US123" and info.title is None
