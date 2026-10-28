import json

import pytest

from ragbench.goldset.schema import Candidate, QuestionType
from ragbench.goldset.split import split_verified

TYPES: list[QuestionType] = ["factual", "how_to", "why_when"]
# 4 factual, 3 how_to, 3 why_when = 10 items
COUNTS = {"factual": 4, "how_to": 3, "why_when": 3}


def _make_candidates() -> list[Candidate]:
    items = []
    n = 0
    for qtype in TYPES:
        for _ in range(COUNTS[qtype]):
            n += 1
            items.append(
                Candidate(
                    qid=f"q{n:02d}",
                    question=f"question {n}?",
                    reference_answer=f"answer {n}",
                    gold_sections=[f"doc{n}.md#s"],
                    gold_spans=[{"doc_id": f"doc{n}.md", "start": 0, "end": 10}],
                    question_type=qtype,
                    source_doc=f"doc{n}.md",
                    verified=True,
                )
            )
    return items


def _write_verified(path, candidates):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for c in candidates:
            handle.write(c.model_dump_json() + "\n")


def test_split_sizes_stratified_and_written(tmp_path):
    verified = tmp_path / "verified.jsonl"
    out = tmp_path / "splits.json"
    _write_verified(verified, _make_candidates())
    result = split_verified(verified_path=verified, out_path=out)
    assert set(result) == {"seed", "dev", "test"}
    assert result["seed"] == 42
    all_qids = {c.qid for c in _make_candidates()}
    dev = set(result["dev"])
    test = set(result["test"])
    assert dev | test == all_qids
    assert dev.isdisjoint(test)
    assert len(dev) == 3
    assert len(test) == 7
    qtype = {c.qid: c.question_type for c in _make_candidates()}
    dev_per_type: dict[str, int] = {}
    for qid in dev:
        dev_per_type[qtype[qid]] = dev_per_type.get(qtype[qid], 0) + 1
    assert dev_per_type == {"factual": 1, "how_to": 1, "why_when": 1}
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == result


def test_split_is_deterministic_for_same_seed(tmp_path):
    verified = tmp_path / "verified.jsonl"
    _write_verified(verified, _make_candidates())
    first = split_verified(verified_path=verified, out_path=tmp_path / "s1.json", seed=42)
    second = split_verified(verified_path=verified, out_path=tmp_path / "s2.json", seed=42)
    assert first == second


def test_split_input_order_does_not_change_result(tmp_path):
    verified = tmp_path / "verified.jsonl"
    _write_verified(verified, _make_candidates())
    forward = split_verified(verified_path=verified, out_path=tmp_path / "f.json")
    verified.write_text("", encoding="utf-8")
    _write_verified(verified, list(reversed(_make_candidates())))
    reverse = split_verified(verified_path=verified, out_path=tmp_path / "r.json")
    assert forward == reverse


def test_split_rejects_empty_verified(tmp_path):
    verified = tmp_path / "verified.jsonl"
    verified.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no verified"):
        split_verified(verified_path=verified, out_path=tmp_path / "s.json")
