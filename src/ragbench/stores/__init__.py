"""Store backends implementing the :class:`~ragbench.types.Store` protocol (Task 2.5)."""

from __future__ import annotations

from ragbench.stores.pgvector_store import PgvectorStore

__all__ = ["PgvectorStore"]
