"""Deterministic tests for the tool parsers and RAG helpers. No network, no model downloads."""

import numpy as np
import pytest

from src.gta.rag import retrieval
from src.gta.rag.chunking import chunk_abstract
from src.gta.tools import gene_lookup as gl
from src.gta.tools import literature_search as ls
from src.gta.tools.pubmed_search import NO_ABSTRACT, _parse_pubmed_xml

# --- chunking -------------------------------------------------------------


def test_short_abstract_is_a_single_unchanged_chunk():
    text = "A short abstract. " * 10
    assert chunk_abstract(text) == [text]


def test_long_abstract_is_windowed_with_overlap_and_loses_nothing():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_abstract(" ".join(words), max_words=200, overlap_words=40)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 200 for c in chunks)
    # consecutive chunks share exactly the overlap
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    # every word appears in at least one chunk
    assert set(" ".join(chunks).split()) == set(words)


def test_empty_abstract_does_not_crash():
    assert chunk_abstract("") == [""]


# --- PubMed XML parsing ---------------------------------------------------

XML = """<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111111</PMID><Article>
  <Journal><Title>J Test</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
  <ArticleTitle>Risk for carriers of <i>BRCA1</i> and <i>BRCA2</i> variants.</ArticleTitle>
  <Abstract>
    <AbstractText Label="BACKGROUND">Unsure whether <i>BRCA1</i> carriers differ.</AbstractText>
    <AbstractText Label="RESULTS">Risk rose (HR 1.4, <i>P</i> = .02).</AbstractText>
  </Abstract></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222222</PMID><Article>
  <Journal><Title>J Other</Title></Journal><ArticleTitle>No abstract here.</ArticleTitle>
</Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_nested_tags_do_not_truncate_title_or_abstract():
    """Regression: .text silently dropped everything after a nested <i> tag."""
    first = _parse_pubmed_xml(XML)[0]
    assert first["title"] == "Risk for carriers of BRCA1 and BRCA2 variants."
    assert "BRCA1 carriers differ" in first["abstract"]
    assert "P = .02" in first["abstract"]


def test_structured_abstract_sections_are_joined_and_fields_parsed():
    first = _parse_pubmed_xml(XML)[0]
    assert first["pmid"] == "111111"
    assert first["journal"] == "J Test" and first["year"] == "2024"
    assert first["abstract"].index("Unsure") < first["abstract"].index("Risk rose")


def test_missing_abstract_gets_placeholder():
    second = _parse_pubmed_xml(XML)[1]
    assert second["abstract"] == "(no abstract available)"


# --- retrieval (embedder replaced with fixed vectors) ---------------------


def _fake_embed(vectors):
    def embed(texts):
        return np.array([vectors[t] for t in texts], dtype=float)

    return embed


def test_retrieval_orders_by_similarity_and_applies_k_and_threshold(monkeypatch):
    vectors = {
        "query": np.array([1.0, 0.0]),
        "close": np.array([0.9, 0.436]) / np.linalg.norm([0.9, 0.436]),
        "mid": np.array([0.6, 0.8]),
        "far": np.array([0.0, 1.0]),
    }
    monkeypatch.setattr(retrieval, "embed", _fake_embed(vectors))
    chunks = [{"text": "far"}, {"text": "close"}, {"text": "mid"}]

    ranked = retrieval.retrieve_top_k("query", chunks, k=5, min_similarity=0.3)
    assert [c["text"] for c in ranked] == ["close", "mid"]  # 'far' (sim 0) is dropped, not padded in
    assert ranked[0]["similarity"] > ranked[1]["similarity"]
    assert len(retrieval.retrieve_top_k("query", chunks, k=1, min_similarity=0.3)) == 1


def test_retrieval_returns_nothing_when_nothing_is_relevant(monkeypatch):
    vectors = {"query": np.array([1.0, 0.0]), "far": np.array([0.0, 1.0])}
    monkeypatch.setattr(retrieval, "embed", _fake_embed(vectors))
    assert retrieval.retrieve_top_k("query", [{"text": "far"}], min_similarity=0.3) == []


def test_retrieval_handles_no_chunks():
    assert retrieval.retrieve_top_k("query", []) == []


# --- input validation lives in the tool functions (agent path AND MCP path) ---

class _Resp:
    def __init__(self, hits):
        self._hits = hits

    def raise_for_status(self):
        pass

    def json(self):
        return {"hits": self._hits}


def _no_network(*args, **kwargs):
    raise AssertionError("the network must not be reached for invalid input")


@pytest.mark.parametrize("bad", ["BRCA*", "TP53 OR BRCA1", "symbol:*", "", "   ", "A" * 31, 123, None])
def test_gene_lookup_rejects_invalid_symbols_before_any_request(monkeypatch, bad):
    monkeypatch.setattr(gl.requests, "get", _no_network)
    result = gl.gene_lookup(bad)
    assert result["found"] is False and "not a valid gene symbol" in result["error"]


def test_gene_lookup_only_accepts_the_requested_gene(monkeypatch):
    """MyGene.info ranks loosely; a different gene coming back must not be passed off as the answer."""
    monkeypatch.setattr(gl.requests, "get", lambda *a, **k: _Resp([{"symbol": "TP53", "name": "tumor protein p53"}]))
    wrong = gl.gene_lookup("Trp53")
    assert wrong["found"] is False and "closest match was 'TP53'" in wrong["error"]


def test_gene_lookup_symbol_match_ignores_case_and_surrounding_whitespace(monkeypatch):
    seen = {}

    def fake_get(url, params, timeout):
        seen["q"] = params["q"]
        return _Resp([{"symbol": "TP53", "name": "tumor protein p53"}])

    monkeypatch.setattr(gl.requests, "get", fake_get)
    result = gl.gene_lookup(" tp53 ")
    assert result["found"] is True and result["symbol"] == "TP53"
    assert seen["q"] == "symbol:tp53"  # stripped before it reaches the query string


@pytest.mark.parametrize(
    "kwargs",
    [{"query": "hi"}, {"query": "x" * 301}, {"query": "BRCA1 risk", "k": 0}, {"query": "BRCA1 risk", "k": 500},
     {"query": "BRCA1 risk", "k": "5"}, {"query": "BRCA1 risk", "k": True}, {"query": "BRCA1 risk", "fetch_n": 5000}],
)
def test_literature_search_rejects_invalid_arguments_before_any_request(monkeypatch, kwargs):
    monkeypatch.setattr(ls, "pubmed_search", _no_network)
    result = ls.literature_search(**kwargs)
    assert result["found"] is False and result["error"].startswith("Invalid arguments")


def test_literature_search_valid_arguments_reach_pubmed(monkeypatch):
    monkeypatch.setattr(ls, "pubmed_search", lambda query, max_results: {"found": False, "error": "no results"})
    assert ls.literature_search("BRCA1 hormone risk", k=3)["error"] == "no results"


# --- small-to-big retrieval ------------------------------------------------


def test_group_by_keeps_best_chunk_per_group_and_k_counts_groups(monkeypatch):
    vectors = {
        "query": np.array([1.0, 0.0]),
        "a1": np.array([0.6, 0.8]),  # similarity 0.6
        "a2": np.array([1.0, 0.0]),  # similarity 1.0  (best chunk of group A)
        "b1": np.array([0.8, 0.6]),  # similarity 0.8
    }
    monkeypatch.setattr(retrieval, "embed", _fake_embed(vectors))
    chunks = [{"text": "a1", "g": "A"}, {"text": "a2", "g": "A"}, {"text": "b1", "g": "B"}]

    ranked = retrieval.retrieve_top_k("query", chunks, k=2, group_by="g")
    assert [(c["g"], c["text"]) for c in ranked] == [("A", "a2"), ("B", "b1")]
    assert len(retrieval.retrieve_top_k("query", chunks, k=1, group_by="g")) == 1
    assert len(retrieval.retrieve_top_k("query", chunks, k=3)) == 3  # without grouping, k counts chunks


def test_literature_search_scores_by_best_window_but_returns_the_whole_abstract(monkeypatch):
    seen = []

    def fake_embed(texts):
        seen.extend(texts)
        return np.array([[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(retrieval, "embed", fake_embed)
    filler = " ".join(f"w{i}" for i in range(200))
    papers = [
        # relevant text in the FIRST and LAST windows: without per-paper grouping this paper would appear twice
        {"pmid": "1", "title": "Paper A", "journal": "J", "year": "2024", "abstract": "alpha opening " + filler + " alpha finding"},
        {"pmid": "2", "title": "Paper B", "journal": "J", "year": "2023", "abstract": filler},
        {"pmid": "3", "title": "Paper C", "journal": "J", "year": "2022", "abstract": NO_ABSTRACT},
    ]
    monkeypatch.setattr(ls, "pubmed_search", lambda query, max_results: {"found": True, "papers": papers})

    out = ls.literature_search("alpha question", k=5)

    assert [r["pmid"] for r in out["results"]] == ["1"]  # B is irrelevant, C has no abstract
    assert out["results"][0]["abstract"] == papers[0]["abstract"]  # whole abstract, not the matching window
    assert out["candidates_fetched"] == 3
    embedded = [t for t in seen if t != "alpha question"]
    assert embedded and all(t.startswith(("Paper A", "Paper B")) for t in embedded)  # titles prepended, C skipped
