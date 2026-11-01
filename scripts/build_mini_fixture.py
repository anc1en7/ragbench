"""Build the committed mini fixture from the verified goldset + corpus (Task 2.10).

Deterministic, offline, stdlib-only. Sources (verbatim copies — no invented text):

- ``data/goldset/verified.jsonl`` — 157 human-verified questions (full schema incl.
  ``gold_spans``; Phase 3's span scorer needs them).
- ``data/documents.jsonl`` — the pinned FastAPI-docs corpus.

Selection algorithm (controller rulings 2–3; no randomness anywhere, so re-running
produces byte-identical files — ``main()`` self-checks this by regenerating and
comparing, printing ``byte-identical: True``):

1. **Documents (exactly 30):** rank every doc_id appearing as ``source_doc`` in the
   verified set by (verified-question count DESC, doc_id ASC); take the first 30.
   Rationale: documents that anchor the most human-verified questions give the mini
   fixture the densest evaluation signal per indexed document.
2. **Questions (exactly 40):** from verified questions whose ``source_doc`` is in the
   30-doc set, group by ``question_type`` ASC; within each type sort by ``qid`` ASC;
   allocate ``floor(40 * type_count / total_available)`` per type (proportional
   stratification preserves the goldset's type mix); distribute the remainder to
   types in alphabetical order (one each) until 40; take the first N of each type's
   sorted list. Alphabetical remainder distribution is an arbitrary but stable tie
   rule; qid ASC makes picks within a type reproducible.

Output (the only writes this script performs):

- ``data/fixtures/mini/documents.jsonl`` — the 30 corpus records verbatim, in
  ranking order.
- ``data/fixtures/mini/questions.jsonl`` — the 40 verified records verbatim, grouped
  by type ASC then qid ASC.

Run via ``uv run python scripts/build_mini_fixture.py``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_PATH = REPO_ROOT / "data" / "goldset" / "verified.jsonl"
DOCUMENTS_PATH = REPO_ROOT / "data" / "documents.jsonl"
OUT_DIR = REPO_ROOT / "data" / "fixtures" / "mini"

N_DOCS = 30
N_QUESTIONS = 40


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required source file missing: {path}")
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {lineno} of {path}: {exc}") from exc
    if not records:
        raise ValueError(f"source file has no records: {path}")
    return records


def _dumps_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    """Canonical JSONL: sorted keys, UTF-8, LF newlines — byte-stable across runs."""
    lines = [
        json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for rec in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def select_docs(verified: list[dict[str, Any]]) -> list[str]:
    """Top-30 doc_ids by (verified-question count DESC, doc_id ASC) — ruling 2."""
    counts = Counter(str(rec["source_doc"]) for rec in verified)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < N_DOCS:
        raise ValueError(
            f"need at least {N_DOCS} source docs in verified.jsonl, found {len(ranked)}"
        )
    doc_ids = [doc_id for doc_id, _ in ranked[:N_DOCS]]
    assert len(doc_ids) == N_DOCS
    return doc_ids


def select_questions(
    verified: list[dict[str, Any]], doc_ids: list[str], target: int = N_QUESTIONS
) -> list[dict[str, Any]]:
    """Proportional-by-type stratified pick of exactly ``target`` questions — ruling 3."""
    doc_set = set(doc_ids)
    available = [rec for rec in verified if rec["source_doc"] in doc_set]
    if len(available) < target:
        raise ValueError(
            f"only {len(available)} verified questions in the {N_DOCS}-doc set, need {target}"
        )
    by_type: dict[str, list[dict[str, Any]]] = {}
    for rec in sorted(available, key=lambda r: (str(r["question_type"]), str(r["qid"]))):
        by_type.setdefault(str(rec["question_type"]), []).append(rec)
    total = len(available)
    alloc = {t: (target * len(recs)) // total for t, recs in sorted(by_type.items())}
    remainder = target - sum(alloc.values())
    for t in sorted(alloc):
        if remainder <= 0:
            break
        alloc[t] += 1
        remainder -= 1
    selection = [rec for t in sorted(by_type) for rec in by_type[t][: alloc[t]]]
    assert len(selection) == target
    return selection


def build_fixture() -> dict[str, bytes]:
    """Compute both fixture files' bytes from the sources (pure, deterministic)."""
    verified = _load_jsonl(VERIFIED_PATH)
    corpus = _load_jsonl(DOCUMENTS_PATH)
    doc_ids = select_docs(verified)
    corpus_by_id = {str(rec["doc_id"]): rec for rec in corpus}
    missing = [doc_id for doc_id in doc_ids if doc_id not in corpus_by_id]
    if missing:
        raise ValueError(f"source docs absent from documents.jsonl: {missing}")
    documents = [corpus_by_id[doc_id] for doc_id in doc_ids]
    questions = select_questions(verified, doc_ids)
    return {
        "documents.jsonl": _dumps_jsonl(documents),
        "questions.jsonl": _dumps_jsonl(questions),
    }


def main() -> None:
    files = build_fixture()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (OUT_DIR / name).write_bytes(payload)

    # Double-run check (ruling 5): regenerate and compare against the written bytes.
    regen = build_fixture()
    identical = regen == files and all(
        (OUT_DIR / name).read_bytes() == payload for name, payload in files.items()
    )
    print(f"byte-identical: {identical}")
    if not identical:
        raise SystemExit("fixture generation is not deterministic — aborting")

    n_docs = files["documents.jsonl"].count(b"\n")
    n_questions = files["questions.jsonl"].count(b"\n")
    print(f"documents={n_docs} questions={n_questions} -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
