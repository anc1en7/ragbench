"""Build ``data/documents.jsonl`` from the fetched raw corpus."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ragbench.corpus.clean import clean_markdown
from ragbench.corpus.sections import parse_sections
from ragbench.types import Document


def build_documents(
    raw_dir: Path = Path("data/raw"),
    out: Path = Path("data/documents.jsonl"),
) -> list[Document]:
    """Parse every markdown file under ``raw_dir`` and write the JSONL corpus."""
    documents: list[Document] = []
    for path in sorted(raw_dir.rglob("*.md")):
        if not path.is_file():
            continue
        doc_id = path.relative_to(raw_dir).as_posix()
        text = clean_markdown(path.read_text(encoding="utf-8"))
        title, sections = parse_sections(doc_id, text)
        documents.append(
            Document(
                doc_id=doc_id,
                title=title or path.stem,
                text=text,
                sections=sections,
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return documents
