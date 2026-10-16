def chunk_abstract(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Split abstract text into overlapping word-windows, but only if needed.

    Most PubMed abstracts are short enough to stay under max_words, in which
    case this returns the text unchanged as a single chunk -- the abstract
    itself is the natural retrieval unit, not an arbitrary token window.
    Splitting only kicks in for unusually long structured abstracts, and the
    overlap ensures a claim near a window boundary keeps enough surrounding
    context to still make sense as a standalone chunk.
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
