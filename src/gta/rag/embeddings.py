from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, ~80MB, fast on CPU

# Loading the model from disk takes a second or two. Lazy-load once per
# process into a module-level singleton rather than reloading on every call.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]):
    """Embed a list of strings into an (n, 384) array of unit-normalized vectors.

    Unit-normalizing here (rather than at comparison time) means cosine
    similarity between any two embeddings reduces to a plain dot product.
    """
    return _get_model().encode(texts, normalize_embeddings=True)
