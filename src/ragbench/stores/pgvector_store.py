"""pgvector Store backend: HNSW cosine dense search + Postgres FTS sparse search (Task 2.5).

Controller rulings (2-8): one physical table per namespace (``chunks_<sanitized>``),
because HNSW needs a fixed-dim typed column while dims differ per embedder; no
``metadata`` column (Chunks round-trip ``metadata={}``); dense score is
``1.0 - cosine_distance``, sparse score is ``ts_rank_cd``; ``size_bytes`` is
``pg_total_relation_size`` (0 when the namespace table is absent).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from ragbench.types import Chunk, Hit

DEFAULT_PG_DSN = "postgresql://ragbench:ragbench@localhost:5432/ragbench"
"""docker-compose default: resolution order is explicit arg → ``PG_DSN`` env → this."""

_FIXED_DIM_RE = re.compile(r"vector\((\d+)\)")


def _table_name(namespace: str) -> str:
    """Table-per-namespace name (ruling 2): ``chunks_<sanitized>``.

    Sanitization: lowercase, every ``[^a-z0-9_]`` becomes ``_``; a segment that
    is empty or starts with a digit gets an ``n_`` prefix so the identifier can
    never start with a digit.
    """
    segment = re.sub(r"[^a-z0-9_]", "_", namespace.lower())
    if not segment or segment[0].isdigit():
        segment = f"n_{segment}"
    return f"chunks_{segment}"


def _namespace_dim(conn: psycopg.Connection[dict[str, Any]], table: str) -> int:
    """Fixed vector dimension of the namespace table's ``embedding`` column.

    Reads the catalog (``format_type`` → ``vector(<dim>)``) so ``add`` validates
    against the table's actual dim even when this instance never called ``reset``.
    """
    row = conn.execute(
        "SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type "
        "FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding' "
        "AND NOT a.attisdropped",
        (table,),
    ).fetchone()
    if row is None:
        raise ValueError(f"namespace table {table!r} does not exist — call reset() first")
    match = _FIXED_DIM_RE.fullmatch(str(row["embedding_type"]))
    if match is None:
        raise ValueError(
            f"embedding column of {table!r} is not a fixed-dim vector: {row['embedding_type']!r}"
        )
    return int(match.group(1))


class PgvectorStore:
    """Postgres + pgvector implementation of the :class:`~ragbench.types.Store` protocol.

    One lazily-opened psycopg connection per instance (autocommit; multi-statement
    writes run inside ``conn.transaction()`` blocks). ``close()`` releases the
    connection and is not part of the ``Store`` protocol.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn: str = dsn or os.environ.get("PG_DSN") or DEFAULT_PG_DSN
        self._conn: psycopg.Connection[dict[str, Any]] | None = None

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        if self._conn is not None and not self._conn.closed:
            return self._conn
        try:
            conn = psycopg.connect(self._dsn, row_factory=dict_row, autocommit=True)
        except psycopg.OperationalError as exc:
            raise RuntimeError(
                f"could not connect to Postgres (PG_DSN={self._dsn!r}); "
                "check PG_DSN or run `make up`"
            ) from exc
        # register_vector() raises unless the extension exists; creating it here
        # (idempotent) keeps first use on a fresh database working. reset() also
        # runs CREATE EXTENSION per ruling 2.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the underlying connection (idempotent; not part of ``Store``)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def reset(self, namespace: str, dim: int) -> None:
        """Drop and recreate the namespace table with HNSW + FTS indexes (idempotent)."""
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        table = sql.Identifier(_table_name(namespace))
        conn = self._connect()
        with conn.transaction():
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
            conn.execute(
                sql.SQL(
                    "CREATE TABLE {} ("
                    "namespace text NOT NULL, chunk_id text NOT NULL, doc_id text NOT NULL, "
                    '"start" integer NOT NULL, "end" integer NOT NULL, text text NOT NULL, '
                    f"embedding vector({dim})"
                    ")"
                ).format(table)
            )
            conn.execute(
                sql.SQL(
                    "CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                ).format(table)
            )
            conn.execute(
                sql.SQL("CREATE INDEX ON {} USING gin (to_tsvector('english', text))").format(table)
            )

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Bulk-insert chunks and their float32 vectors (validates count and dim)."""
        table_name = _table_name(namespace)
        conn = self._connect()
        dim = _namespace_dim(conn, table_name)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n, dim); got shape {vectors.shape}")
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"len(chunks)={len(chunks)} != vectors.shape[0]={vectors.shape[0]}")
        if vectors.shape[1] != dim:
            raise ValueError(
                f"vectors.shape[1]={vectors.shape[1]} != namespace dim {dim} "
                f"for namespace {namespace!r}"
            )
        if not chunks:
            return
        rows = np.asarray(vectors, dtype=np.float32)
        args = [
            (
                namespace,
                chunk.chunk_id,
                chunk.doc_id,
                chunk.start,
                chunk.end,
                chunk.text,
                Vector(row.tolist()),
            )
            for chunk, row in zip(chunks, rows, strict=True)
        ]
        insert = sql.SQL(
            'INSERT INTO {} (namespace, chunk_id, doc_id, "start", "end", text, embedding) '
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        ).format(sql.Identifier(table_name))
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(insert, args)

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]:
        """Top-k cosine neighbors: score = ``1.0 - cosine_distance``, descending."""
        if k <= 0:
            return []
        conn = self._connect()
        stmt = sql.SQL(
            'SELECT chunk_id, doc_id, "start", "end", text, embedding <=> %s AS distance '
            "FROM {} ORDER BY embedding <=> %s LIMIT %s"
        ).format(sql.Identifier(_table_name(namespace)))
        vec = Vector(np.asarray(qvec, dtype=np.float32))
        try:
            # Every HNSW query runs with ef_search=100 (session GUC; ruling 2).
            conn.execute("SET hnsw.ef_search = 100")
            rows = conn.execute(stmt, (vec, vec, k)).fetchall()
        except psycopg.errors.UndefinedTable:
            conn.rollback()  # no-op under autocommit; keeps the session clean
            return []
        return [
            Hit(
                chunk=Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    start=row["start"],
                    end=row["end"],
                    text=row["text"],
                    metadata={},
                ),
                score=1.0 - float(row["distance"]),
                source="dense",
            )
            for row in rows
        ]

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]:
        """Top-k FTS matches: ``websearch_to_tsquery`` + ``ts_rank_cd``, descending.

        Empty or stopword-only queries yield an empty tsquery and therefore ``[]``
        (never an error).
        """
        if k <= 0 or not query.strip():
            return []
        conn = self._connect()
        stmt = sql.SQL(
            'SELECT chunk_id, doc_id, "start", "end", text, '
            "ts_rank_cd(to_tsvector('english', text), "
            "websearch_to_tsquery('english', %s)) AS score "
            "FROM {} "
            "WHERE to_tsvector('english', text) @@ websearch_to_tsquery('english', %s) "
            "ORDER BY score DESC "
            "LIMIT %s"
        ).format(sql.Identifier(_table_name(namespace)))
        try:
            rows = conn.execute(stmt, (query, query, k)).fetchall()
        except psycopg.errors.UndefinedTable:
            conn.rollback()
            return []
        return [
            Hit(
                chunk=Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    start=row["start"],
                    end=row["end"],
                    text=row["text"],
                    metadata={},
                ),
                score=float(row["score"]),
                source="sparse",
            )
            for row in rows
        ]

    def size_bytes(self, namespace: str) -> int:
        """Total relation size of the namespace table (0 when absent)."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(pg_total_relation_size(to_regclass(%s)), 0) AS size",
            (_table_name(namespace),),
        ).fetchone()
        assert row is not None  # SELECT with COALESCE always returns one row
        return int(row["size"])
