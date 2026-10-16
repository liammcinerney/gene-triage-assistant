from src.vta.rag.embeddings import embed


def retrieve_top_k(
    query: str, chunks: list[dict], k: int = 5, min_similarity: float = 0.3
) -> list[dict]:
    """Return the k chunks most semantically similar to the query.

    Args:
        query: The text to compare chunks against.
        chunks: List of dicts, each with a "text" key plus arbitrary metadata.
        k: Max number of chunks to return.
        min_similarity: Chunks scoring below this are dropped entirely,
            rather than padding the result out to k with weak matches --
            "nothing here is actually relevant" is a real, useful answer,
            not something to paper over.

    Returns:
        The input dicts, filtered and sorted by similarity descending, each
        with a "similarity" field added.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    chunk_vectors = embed(texts)  # (n, 384), unit-normalized
    query_vector = embed([query])[0]  # (384,), unit-normalized

    # Unit-normalized vectors -> cosine similarity is just the dot product.
    similarities = chunk_vectors @ query_vector

    scored = [
        {**chunk, "similarity": float(sim)}
        for chunk, sim in zip(chunks, similarities)
    ]
    scored = [c for c in scored if c["similarity"] >= min_similarity]
    scored.sort(key=lambda c: c["similarity"], reverse=True)
    return scored[:k]
