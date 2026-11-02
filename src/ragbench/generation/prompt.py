"""Versioned grounded-answer prompt (Task 4.3).

``PROMPT_VERSION`` is recorded in every run's generation block; change it whenever the
prompt text changes so generation metrics from different prompts are never mixed.
Contexts are numbered in retrieval order. The model is told to answer only from them
and to say so when they do not contain the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

PROMPT_VERSION = "gen-v1"

SYSTEM_PROMPT = (
    "You answer questions about the FastAPI documentation. Use ONLY the numbered context "
    "passages provided. If they do not contain the answer, reply exactly: "
    '"The provided context does not contain the answer." Be concise and factual.'
)


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...


def build_messages(question: str, contexts: Sequence[str]) -> list[dict[str, str]]:
    """System + user messages with numbered context passages."""
    passages = "\n\n".join(f"[{i}] {text.strip()}" for i, text in enumerate(contexts, start=1))
    user = f"Context:\n{passages}\n\nQuestion: {question.strip()}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate_answer(
    client: ChatClient,
    question: str,
    contexts: Sequence[str],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Generate a grounded answer for ``question`` from ``contexts``."""
    reply = client.chat(
        build_messages(question, contexts), temperature=temperature, max_tokens=max_tokens
    )
    return reply.strip()
