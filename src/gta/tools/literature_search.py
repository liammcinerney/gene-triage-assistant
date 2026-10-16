from src.gta.rag.chunking import chunk_abstract
from src.gta.rag.retrieval import retrieve_top_k
from src.gta.tools.pubmed_search import NO_ABSTRACT, pubmed_search

# Bounds live here, enforced by the function itself, so every caller (agent
# loop, MCP server, eval harness) gets the same behaviour. The MCP schema
# imports these constants.
MIN_QUERY_LEN, MAX_QUERY_LEN = 3, 300
MIN_K, MAX_K = 1, 10
MAX_FETCH_N = 50


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def literature_search(query: str, k: int = 5, fetch_n: int = 15) -> dict:
    """Find the most relevant papers for a specific question, grounded in PubMed.

    Two-stage RAG pipeline: pubmed_search does coarse lexical candidate
    generation (PubMed's own keyword relevance ranking), then embedding
    similarity reranks and filters that pool down to what's actually
    relevant to `query` before any of it reaches an LLM's context.

    Args:
        query: The specific question or claim to find evidence for.
        k: Max number of papers to return after semantic filtering.
        fetch_n: How many candidate papers to fetch from PubMed before filtering.

    Returns:
        A JSON-serializable dict. On success, "found" is True and "results"
        is a list of {pmid, title, journal, year, abstract, similarity} with the
        FULL abstract, one entry per paper, sorted by relevance. On failure -- no PubMed candidates, or none of
        the candidates were relevant enough -- "found" is False and "error"
        explains why.
    """
    problems = []
    if not isinstance(query, str) or not MIN_QUERY_LEN <= len(query.strip()) <= MAX_QUERY_LEN:
        problems.append(f"query must be text of {MIN_QUERY_LEN}-{MAX_QUERY_LEN} characters")
    if not _is_int(k) or not MIN_K <= k <= MAX_K:
        problems.append(f"k must be an integer from {MIN_K} to {MAX_K}")
    if not _is_int(fetch_n) or not 1 <= fetch_n <= MAX_FETCH_N:
        problems.append(f"fetch_n must be an integer from 1 to {MAX_FETCH_N}")
    if problems:
        return {"found": False, "error": "Invalid arguments: " + "; ".join(problems), "candidates_fetched": 0}

    fetched = pubmed_search(query, max_results=fetch_n)
    if not fetched.get("found"):
        return {
            "found": False,
            "error": fetched.get("error", "no candidates found"),
            "candidates_fetched": 0,
        }

    # Small-to-big retrieval: short windows are the retrieval KEYS (sized to the
    # embedder's input limit), the whole abstract is what we RETURN. The title
    # is prepended to each window so a mid-abstract window still says what the
    # paper is about. Papers without an abstract carry no evidence; skip them.
    papers = {p["pmid"]: p for p in fetched["papers"] if p["abstract"] != NO_ABSTRACT}
    windows = [
        {"text": f"{paper['title']} {piece}", "pmid": pmid}
        for pmid, paper in papers.items()
        for piece in chunk_abstract(paper["abstract"])
    ]

    # A paper is scored by its best window; k counts papers, not windows.
    top = retrieve_top_k(query, windows, k=k, group_by="pmid")
    if not top:
        return {
            "found": False,
            "error": (
                f"{len(papers)} candidate papers with abstracts were fetched from PubMed, "
                f"but none were semantically relevant enough to '{query}'"
            ),
            "candidates_fetched": len(fetched["papers"]),
        }

    return {
        "found": True,
        "candidates_fetched": len(fetched["papers"]),
        "results": [
            {
                "pmid": hit["pmid"],
                "title": papers[hit["pmid"]]["title"],
                "journal": papers[hit["pmid"]]["journal"],
                "year": papers[hit["pmid"]]["year"],
                "abstract": papers[hit["pmid"]]["abstract"],
                "similarity": round(hit["similarity"], 3),
            }
            for hit in top
        ],
    }


if __name__ == "__main__":
    import json

    result = literature_search(
        "does hormone replacement therapy increase breast cancer risk in BRCA1 carriers",
        k=3,
        fetch_n=10,
    )
    print(json.dumps(result, indent=2))
