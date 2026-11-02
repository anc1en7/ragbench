"""Mini-fixture integrity tests (Task 2.10, ruling 4) — offline, committed files only.

Independently reproduces the documented selection algorithm from the brief:

- Doc ranking: all doc_ids appearing as ``source_doc`` in ``data/goldset/verified.jsonl``
  ranked by (verified-question count DESC, doc_id ASC); the fixture must be exactly the
  first 30 of that ranking, written in ranking order.
- Question stratification: from verified questions whose ``source_doc`` is in the 30-doc
  set, group by ``question_type`` ASC, sort each group by ``qid`` ASC, allocate
  ``floor(40 * type_count / total_available)`` per type, distribute the remainder to
  types in alphabetical order (one each) until 40, take the first N of each group.

A corpus/goldset edit that would change the fixture therefore fails loudly here (the
expected ranking and allocation are recomputed from the sources, not read from the
script). Fixture question records must be content-identical (parsed dicts) to their
``verified.jsonl`` counterparts — Phase 3 needs the full schema including gold_spans.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIED_PATH = ROOT / "data" / "goldset" / "verified.jsonl"
DOCUMENTS_PATH = ROOT / "data" / "documents.jsonl"
MINI_DIR = ROOT / "data" / "fixtures" / "mini"
MINI_DOCUMENTS_PATH = MINI_DIR / "documents.jsonl"
MINI_QUESTIONS_PATH = MINI_DIR / "questions.jsonl"

N_DOCS = 30
N_QUESTIONS = 40


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _expected_doc_ranking(verified: list[dict[str, Any]]) -> list[str]:
    """First 30 doc_ids by (verified-question count DESC, doc_id ASC)."""
    counts = Counter(str(rec["source_doc"]) for rec in verified)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assert len(ranked) >= N_DOCS, "goldset must have at least 30 source docs"
    return [doc_id for doc_id, _ in ranked[:N_DOCS]]


def _expected_type_allocation(
    available: list[dict[str, Any]], target: int = N_QUESTIONS
) -> dict[str, int]:
    """Proportional-by-type allocation with alphabetical remainder distribution."""
    by_type = Counter(str(rec["question_type"]) for rec in available)
    total = len(available)
    assert total >= target, "not enough verified questions in the 30-doc set"
    alloc = {t: (target * count) // total for t, count in sorted(by_type.items())}
    remainder = target - sum(alloc.values())
    for t in sorted(alloc):
        if remainder <= 0:
            break
        alloc[t] += 1
        remainder -= 1
    return alloc


def test_documents_fixture_is_top30_ranking():
    """Runs everywhere (CI included): only committed files are read."""
    docs = _load_jsonl(MINI_DOCUMENTS_PATH)
    assert len(docs) == N_DOCS
    doc_ids = [rec["doc_id"] for rec in docs]
    assert len(set(doc_ids)) == N_DOCS, "duplicate doc_ids in mini documents"

    verified = _load_jsonl(VERIFIED_PATH)
    assert doc_ids == _expected_doc_ranking(verified), (
        "fixture docs must be exactly the top-30 (count DESC, doc_id ASC) ranking; "
        "regenerate with scripts/build_mini_fixture.py"
    )


@pytest.mark.skipif(
    not DOCUMENTS_PATH.exists(),
    reason="data/documents.jsonl is gitignored (fetched by `make ingest`); verbatim check is local",
)
def test_documents_fixture_records_are_verbatim_corpus_copies():
    corpus_by_id = {rec["doc_id"]: rec for rec in _load_jsonl(DOCUMENTS_PATH)}
    for rec in _load_jsonl(MINI_DOCUMENTS_PATH):
        assert rec["doc_id"] in corpus_by_id, f"{rec['doc_id']} missing from corpus"
        assert rec == corpus_by_id[rec["doc_id"]], f"{rec['doc_id']} is not a verbatim copy"


def test_questions_fixture_is_stratified_verified_copies():
    questions = _load_jsonl(MINI_QUESTIONS_PATH)
    assert len(questions) == N_QUESTIONS
    qids = [rec["qid"] for rec in questions]
    assert len(set(qids)) == N_QUESTIONS, "duplicate qids in mini questions"

    verified = _load_jsonl(VERIFIED_PATH)
    verified_by_qid = {rec["qid"]: rec for rec in verified}
    mini_doc_ids = {rec["doc_id"] for rec in _load_jsonl(MINI_DOCUMENTS_PATH)}
    for rec in questions:
        assert rec["qid"] in verified_by_qid, f"{rec['qid']} missing from verified.jsonl"
        assert rec == verified_by_qid[rec["qid"]], f"{rec['qid']} is not a verbatim copy"
        assert rec["source_doc"] in mini_doc_ids, f"{rec['source_doc']} not in mini docs"

    expected_docs = set(_expected_doc_ranking(verified))
    available = [rec for rec in verified if rec["source_doc"] in expected_docs]
    expected_alloc = _expected_type_allocation(available)
    actual_counts = dict(Counter(str(rec["question_type"]) for rec in questions))
    assert actual_counts == expected_alloc, (
        "per-type counts must match the documented proportional allocation; "
        "regenerate with scripts/build_mini_fixture.py"
    )
