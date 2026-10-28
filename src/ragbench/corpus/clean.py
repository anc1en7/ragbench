"""Clean raw markdown corpus files (front matter, line endings, blank runs)."""

from __future__ import annotations

from collections.abc import Sequence


def fence_mask(lines: Sequence[str]) -> list[bool]:
    """Mark lines belonging to fenced code blocks (``` or ~~~ fences)."""
    mask = [False] * len(lines)
    inside = False
    marker = ""
    for idx, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if not inside:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                inside = True
                marker = stripped[:3]
                mask[idx] = True
        else:
            mask[idx] = True
            if stripped.startswith(marker):
                inside = False
    return mask


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :])
    return text


def clean_markdown(text: str) -> str:
    """Normalize a raw corpus file into cleaned markdown.

    Strips YAML front matter, normalizes line endings, and collapses runs of
    blank lines outside code fences. Offsets for sections are always computed
    against the cleaned output, so this runs first (spec 5.2).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_front_matter(text)
    lines = text.split("\n")
    mask = fence_mask(lines)
    cleaned: list[str] = []
    blanks = 0
    for line, inside_fence in zip(lines, mask, strict=True):
        if not inside_fence and not line.strip():
            blanks += 1
            if blanks == 1:
                cleaned.append(line)
        else:
            blanks = 0
            cleaned.append(line)
    return "\n".join(cleaned)
