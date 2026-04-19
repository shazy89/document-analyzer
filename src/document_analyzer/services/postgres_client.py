from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from document_analyzer.core.config import Settings

if TYPE_CHECKING:
    import psycopg


# ── SQL statements ───────────────────────────────────────────

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS document_chunks (
    id          TEXT PRIMARY KEY,
    document    TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}'::jsonb,
    source      TEXT,
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', document)) STORED
);
"""

_CREATE_INDEX_TSV = """\
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN (tsv);
"""

_CREATE_INDEX_SOURCE = """\
CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks (source);
"""

_UPSERT = """\
INSERT INTO document_chunks (id, document, metadata, source)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    document = EXCLUDED.document,
    metadata = EXCLUDED.metadata,
    source   = EXCLUDED.source;
"""

_SEARCH = """\
SELECT id, document, metadata,
       ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
FROM document_chunks
WHERE tsv @@ plainto_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s;
"""

_LIST_SOURCES = """\
SELECT DISTINCT source FROM document_chunks WHERE source IS NOT NULL ORDER BY source;
"""

_DELETE_BY_SOURCE = """\
DELETE FROM document_chunks WHERE source = %s;
"""


# ── Service ──────────────────────────────────────────────────


class PostgresService:
    """Thin wrapper around PostgreSQL for BM25/full-text search.

    Follows the same construction pattern as ``ChromaService``:
    primitive-only ``__init__``, ``from_settings`` factory, lazy connection init.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        db: str,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._db = db
        self._conn: psycopg.Connection | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> PostgresService:
        return cls(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
            db=settings.postgres_db,
        )

    # ── public API ───────────────────────────────────────────

    def heartbeat(self) -> bool:
        """Return ``True`` if PostgreSQL is reachable."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def init_schema(self) -> None:
        """Create the ``document_chunks`` table and indexes if they do not exist."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
            cur.execute(_CREATE_INDEX_TSV)
            cur.execute(_CREATE_INDEX_SOURCE)
        conn.commit()

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents into the ``document_chunks`` table."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            for i, doc_id in enumerate(ids):
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                source = meta.get("source") if meta else None
                cur.execute(
                    _UPSERT,
                    (doc_id, documents[i], json.dumps(meta), source),
                )
        conn.commit()

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Full-text search using PostgreSQL ``ts_rank_cd``.

        Returns a list of result dicts with keys
        ``id``, ``document``, ``metadata``, ``score``.
        """
        conn = self._get_connection()
        results: list[dict[str, Any]] = []

        for query_text in query_texts:
            with conn.cursor() as cur:
                cur.execute(_SEARCH, (query_text, query_text, n_results))
                rows = cur.fetchall()
                for row in rows:
                    metadata = row[2] if row[2] else {}
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    results.append(
                        {
                            "id": row[0],
                            "document": row[1],
                            "metadata": metadata,
                            "score": float(row[3]),
                        }
                    )

        return results

    def list_sources(self) -> list[str]:
        """Return distinct source values from all stored chunks."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(_LIST_SOURCES)
            return [row[0] for row in cur.fetchall()]

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks with the given source. Returns the number of rows deleted."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(_DELETE_BY_SOURCE, (source,))
            count = cur.rowcount
        conn.commit()
        return count

    # ── internals ────────────────────────────────────────────

    def _get_connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = self._build_connection()
        return self._conn

    def _build_connection(self) -> psycopg.Connection:
        import psycopg

        return psycopg.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            dbname=self._db,
            autocommit=False,
        )
