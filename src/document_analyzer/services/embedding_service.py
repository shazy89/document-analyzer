from __future__ import annotations

from document_analyzer.models.chunking import DocumentChunk

# TODO: Replace Chroma's default embedding function with a production-grade
#       embedding provider (e.g. Together AI, OpenAI, Cohere) when ready.
#       The default uses the all-MiniLM-L6-v2 ONNX model and runs locally
#       — fine for development but not ideal for production workloads.


class EmbeddingService:
    """Compute embeddings using Chroma's built-in default embedding function.

    This is a free, local embedding function (``all-MiniLM-L6-v2`` via ONNX).
    Swap the ``_get_ef()`` method when migrating to a remote provider.
    """

    def __init__(self) -> None:
        self._ef: object | None = None

    # ── lazy init ────────────────────────────────────────────

    def _get_ef(self):  # noqa: ANN202
        """Return the default Chroma embedding function (lazy)."""
        if self._ef is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._ef = DefaultEmbeddingFunction()
        return self._ef

    # ── public API ───────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ef = self._get_ef()
        raw = ef(texts)
        
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in raw]

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Attach embedding vectors to a list of ``DocumentChunk`` objects.

        Mutates the chunks in-place **and** returns them for convenience.
        """
        if not chunks:
            return chunks

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, vector in zip(chunks, embeddings):
            chunk.embedding = vector

        return chunks
