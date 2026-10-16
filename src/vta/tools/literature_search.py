from src.vta.rag.chunking import chunk_abstract
from src.vta.rag.retrieval import retrieve_top_k
from src.vta.tools.pubmed_search import pubmed_search


def literature_search(query: str, k: int = 5, fetch_n: int = 15) -> dict:
    """Find the most relevant evidence for a specific question, grounded in PubMed.

    Two-stage RAG pipeline: pubmed_search does coarse lexical candidate
    generation (PubMed's own keyword relevance ranking), then embedding
    similarity reranks and filters that pool down to what's actually
    relevant to `query` before any of it reaches an LLM's context.

    Args:
        query: The specific question or claim to find evidence for.
        k: Max number of evidence chunks to return after semantic filtering.
        fetch_n: How many candidate papers to fetch from PubMed before filtering.

    Returns:
        A JSON-serializable dict. On success, "found" is True and "results"
        is a list of {pmid, title, journal, year, chunk_text, similarity},
        sorted by relevance. On failure -- no PubMed candidates, or none of
        the candidates were relevant enough -- "found" is False and "error"
        explains why.
    """
    fetched = pubmed_search(query, max_results=fetch_n)
    if not fetched.get("found"):
        return {
            "found": False,
            "error": fetched.get("error", "no candidates found"),
            "candidates_fetched": 0,
        }

    chunks = []
    for paper in fetched["papers"]:
        for piece in chunk_abstract(paper["abstract"]):
            chunks.append(
                {
                    "text": piece,
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "journal": paper["journal"],
                    "year": paper["year"],
                }
            )

    top = retrieve_top_k(query, chunks, k=k)
    if not top:
        return {
            "found": False,
            "error": (
                f"{len(chunks)} candidate chunks fetched from PubMed, but none "
                f"were semantically relevant enough to '{query}'"
            ),
            "candidates_fetched": len(fetched["papers"]),
        }

    return {
        "found": True,
        "candidates_fetched": len(fetched["papers"]),
        "results": [
            {
                "pmid": c["pmid"],
                "title": c["title"],
                "journal": c["journal"],
                "year": c["year"],
                "chunk_text": c["text"],
                "similarity": round(c["similarity"], 3),
            }
            for c in top
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
