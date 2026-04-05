from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from document_analyzer.core.config import Settings

if TYPE_CHECKING:
    import chromadb


# ── Protocol duck-types for the ChromaDB SDK ────────────────


class _ChromaCollection(Protocol):
    def add(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def query(
        self,
        query_texts: list[str] | None = None,
        n_results: int = 10,
    ) -> dict[str, Any]: ...


class _ChromaClient(Protocol):
    def heartbeat(self) -> int: ...

    def get_or_create_collection(self, name: str) -> _ChromaCollection: ...


# ── Service ──────────────────────────────────────────────────


class ChromaService:
    """Thin wrapper around the ChromaDB HTTP client.

    Follows the same construction pattern as ``TogetherChatService``:
    primitive-only ``__init__``, ``from_settings`` factory, lazy SDK init.
    """

    def __init__(self, *, host: str, port: int, collection_name: str) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client: chromadb.HttpClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ChromaService:
        return cls(
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection_name=settings.chroma_collection,
        )

    # ── public API ───────────────────────────────────────────

    def heartbeat(self) -> bool:
        """Return ``True`` if ChromaDB is reachable."""
        client = self._get_client()
        try:
            client.heartbeat()
            return True
        except Exception:
            return False

    def get_or_create_collection(self) -> _ChromaCollection:
        """Return the configured collection, creating it if needed."""
        client = self._get_client()
        return client.get_or_create_collection(name=self._collection_name)
    
    def list_collections(self) -> list[str]:
        """Return a list of collection names in the ChromaDB instance."""
        client = self._get_client()
        collections = client.list_collections()
        return [c.name for c in collections]
    
    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add (or upsert) documents into the collection."""
        collection = self.get_or_create_collection()
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the collection and return a list of result dicts.

        Each dict has keys ``id``, ``document``, ``metadata``, ``distance``.
        """
        collection = self.get_or_create_collection()
        raw = collection.query(query_texts=query_texts, n_results=n_results)

        results: list[dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            results.append(
                {
                    "id": doc_id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return results

    # ── internals ────────────────────────────────────────────

    def _get_client(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> chromadb.HttpClient:
        import chromadb

        return chromadb.HttpClient(host=self._host, port=self._port)
