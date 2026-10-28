"""Parse cleaned markdown into heading-delimited sections (spec 4 and 5.2).

Sections are non-overlapping: each heading owns the text from its own line up
to the start of the next heading of any level (or end of document), and char
offsets point into the *cleaned* document text.
"""

from __future__ import annotations

import re

from ragbench.corpus.clean import fence_mask
from ragbench.types import Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ANCHOR = re.compile(r"\s*\{\s*#([^}\s]+)\s*\}\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

Heading = tuple[int, int, str, str]  # level, line index, title, slug


def slugify(text: str) -> str:
    slug = _NON_ALNUM.sub("-", text.lower()).strip("-")
    return slug or "section"


def _split_heading(match: re.Match[str]) -> tuple[int, str, str]:
    level = len(match.group(1))
    body = match.group(2)
    anchor = _ANCHOR.search(body)
    if anchor:
        title = body[: anchor.start()].rstrip()
        slug = anchor.group(1).lower()
    else:
        title = body
        slug = slugify(title)
    return level, title, slug


def parse_sections(doc_id: str, text: str) -> tuple[str, list[Section]]:
    """Return ``(document title, sections)`` for cleaned markdown ``text``."""
    lines = text.split("\n")
    mask = fence_mask(lines)

    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    headings: list[Heading] = []
    for idx, line in enumerate(lines):
        if mask[idx]:
            continue
        match = _HEADING.match(line)
        if match is None:
            continue
        level, title, slug = _split_heading(match)
        headings.append((level, idx, title, slug))

    if not headings:
        return "", []

    document_title = headings[0][2]
    stack: list[tuple[int, str]] = []
    used_slugs: set[str] = set()
    sections: list[Section] = []

    for i, (level, idx, title, slug) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        unique = slug
        if unique in used_slugs:
            n = 2
            while f"{slug}-{n}" in used_slugs:
                n += 1
            unique = f"{slug}-{n}"
        used_slugs.add(unique)
        heading_path = " > ".join([*(t for _lvl, t in stack), title])
        start = offsets[idx]
        end = offsets[headings[i + 1][1]] if i + 1 < len(headings) else len(text)
        sections.append(
            Section(
                section_id=f"{doc_id}#{unique}",
                heading_path=heading_path,
                start=start,
                end=end,
            )
        )
        stack.append((level, title))

    return document_title, sections
