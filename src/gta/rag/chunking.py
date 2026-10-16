def chunk_abstract(text: str, max_words: int = 80, overlap_words: int = 20) -> list[str]:
    """Split abstract text into overlapping word windows to use as retrieval keys.

    The window size is set by the embedding model, not by what reads well.
    all-MiniLM-L6-v2 silently ignores everything past 256 word-pieces, and
    biomedical text runs about 1.7 word-pieces per word. Measured on 192 real
    abstracts, 220-word windows overflowed that limit 60% of the time; 80-word
    windows (with the paper title prepended by the caller) overflow 0.7% of the
    time. A window that overflows is ranked without its tail, which is often
    the results section.

    These windows are only used to FIND papers. literature_search returns the
    whole abstract of each matching paper, so no reader ever sees a fragment.
    The overlap keeps a claim near a boundary intact in at least one window.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks
