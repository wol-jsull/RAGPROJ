import pytest

from patent_rag.citations import extract_citations
from patent_rag.errors import InvalidQueryError
from patent_rag.validation import validate_query


@pytest.mark.parametrize("bad", [None, "", "   ", "\n\t", "?!", "ab"])
def test_rejects_empty_short_or_symbol_only_queries(bad):
    with pytest.raises(InvalidQueryError) as exc:
        validate_query(bad)
    assert exc.value.user_message  # always a displayable message


def test_rejects_overlong_query():
    with pytest.raises(InvalidQueryError, match="too long"):
        validate_query("a" * 501, max_chars=500)


def test_normalizes_whitespace_and_strips_control_characters():
    assert validate_query("  What\x00 is\n\n  attention?\x07 ") == "What is attention?"


def test_citations_split_valid_and_invalid_and_deduplicate():
    report = extract_citations("Claim one [S1]. Claim two [S3][s1]. Bogus [S9] and [S0].", num_sources=3)
    assert report.cited == [1, 3]
    assert report.invalid == [9, 0]
    assert report.has_citations


def test_no_citations():
    report = extract_citations("An answer with no sources.", num_sources=5)
    assert not report.has_citations and report.invalid == []
