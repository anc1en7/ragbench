"""Pure quality filters for gold-set candidates (spec 5.6 step 4).

Thresholds are module constants and are recorded in generation stats metadata.
"""

from __future__ import annotations

import re

import numpy as np

MAX_HEADING_COPY_RATIO = 0.6
MAX_LEXICAL_OVERLAP = 0.6
MAX_CODE_FENCE_RATIO = 0.6
DEDUP_THRESHOLD = 0.92

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def heading_copy_ratio(question: str, heading: str) -> float:
    """Fraction of the question's tokens that appear in the heading.

    High ratio => the question near-copies the heading text.
    """
    q_tokens = _tokens(question)
    if not q_tokens:
        return 0.0
    h_tokens = set(_tokens(heading))
    if not h_tokens:
        return 0.0
    return sum(1 for token in q_tokens if token in h_tokens) / len(q_tokens)


def lexical_overlap(question: str, section_text: str) -> float:
    """Longest run of consecutive question tokens appearing verbatim in the section.

    High ratio => the question is a near-verbatim copy of section phrasing,
    which would unfairly favor sparse retrieval.
    """
    q_tokens = _tokens(question)
    if not q_tokens:
        return 0.0
    s_tokens = _tokens(section_text)
    if not s_tokens:
        return 0.0
    best = 0
    run = 0
    j = 0
    for token in s_tokens:
        if j < len(q_tokens) and token == q_tokens[j]:
            run += 1
            j += 1
            best = max(best, run)
        else:
            run = 0
            j = 0
    return best / len(q_tokens)


def dedup(embeddings: np.ndarray, threshold: float = DEDUP_THRESHOLD) -> set[int]:
    """Greedy near-duplicate removal. Returns indices to DROP.

    Keeps the earliest item; drops any later item whose cosine similarity to an
    already-kept item is >= threshold. Vectors need not be normalized.
    """
    dropped: set[int] = set()
    kept: list[int] = []
    if embeddings.size == 0:
        return dropped
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    for i in range(len(embeddings)):
        if kept:
            sims = normed[kept] @ normed[i]
            if float(np.max(sims)) >= threshold:
                dropped.add(i)
                continue
        kept.append(i)
    return dropped


def code_fence_ratio(text: str) -> float:
    """Fraction of non-empty lines that are inside (or are markers of) code fences."""
    non_empty = 0
    in_fence_lines = 0
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        non_empty += 1
        if stripped.startswith("```"):
            in_fence = not in_fence
            in_fence_lines += 1
        elif in_fence:
            in_fence_lines += 1
    return in_fence_lines / non_empty if non_empty else 0.0
