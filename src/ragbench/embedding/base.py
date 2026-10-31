"""Model-card-verified query/document prefix strings for the supported embedder models
(Task 2.4, ruling 1).

Prefixes are applied inside the embedder's model-call path (ruling 4): the query side
receives the query prefix/instruction, the document side the passage prefix (e5) or
nothing (bge, MiniLM). The strings below were transcribed from the Hugging Face model
cards recorded in ``docs/METHODOLOGY.md`` and are tested as pure functions.
"""

from __future__ import annotations

QUERY_PREFIXES: dict[str, str] = {
    "intfloat/e5-small-v2": "query: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

DOCUMENT_PREFIXES: dict[str, str] = {
    "intfloat/e5-small-v2": "passage: ",
}


def query_prefix(model_name: str) -> str:
    """Query-side prefix/instruction for ``model_name``; empty when the model needs none."""
    return QUERY_PREFIXES.get(model_name, "")


def document_prefix(model_name: str) -> str:
    """Document-side prefix for ``model_name``; empty when the model needs none."""
    return DOCUMENT_PREFIXES.get(model_name, "")
