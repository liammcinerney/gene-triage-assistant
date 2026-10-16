"""Deterministic tests for the tool parsers and RAG helpers. No network, no model downloads."""

import numpy as np

from src.vta.rag import retrieval
from src.vta.rag.chunking import chunk_abstract
from src.vta.tools.pubmed_search import _parse_pubmed_xml

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
