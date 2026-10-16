from src.gta.rag.embeddings import embed


def retrieve_top_k(
    query: str,
    chunks: list[dict],
    k: int = 5,
    min_similarity: float = 0.3,
    group_by: str | None = None,
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
        group_by: If given, a key present on every chunk (e.g. "pmid"). Only
            the best-scoring chunk per distinct value is kept, so k counts
            groups, not chunks. This is what lets small windows act as
            retrieval keys while the caller returns the whole parent document.

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

    if group_by:
        # Sorted best-first, so the first chunk seen per group is its best.
        best = {}
        for chunk in scored:
            best.setdefault(chunk[group_by], chunk)
        scored = list(best.values())

    return scored[:k]
