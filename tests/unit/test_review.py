import io
import json

from rich.console import Console

from ragbench.goldset.review import review
from ragbench.goldset.schema import Candidate

CANDIDATES = [
    Candidate(
        qid="q1",
        question="How do I declare a query parameter?",
        reference_answer="Use Query from fastapi.",
        gold_sections=["guide.md#query"],
        gold_spans=[{"doc_id": "guide.md", "start": 0, "end": 40}],
        question_type="how_to",
        source_doc="guide.md",
    ),
    Candidate(
        qid="q2",
        question="What is a path operation?",
        reference_answer="A function decorated with a router method.",
        gold_sections=["guide.md#path"],
        gold_spans=[{"doc_id": "guide.md", "start": 50, "end": 90}],
        question_type="factual",
        source_doc="guide.md",
    ),
    Candidate(
        qid="q3",
        question="Why use async def?",
        reference_answer="To allow concurrency.",
        gold_sections=["guide.md#async"],
        gold_spans=[{"doc_id": "guide.md", "start": 100, "end": 140}],
        question_type="why_when",
        source_doc="guide.md",
    ),
]


def _write_candidates(path, candidates):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for c in candidates:
            handle.write(c.model_dump_json() + "\n")


def _scripted(*keys):
    it = iter(keys)

    def fn(_prompt=""):
        return next(it)

    return fn


def _quiet_console():
    return Console(file=io.StringIO(), width=100, force_terminal=False)


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_accept_reject_quit_writes_verified_and_log(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, CANDIDATES)
    summary = review(
        candidates_path=candidates,
        verified_path=tmp_path / "verified.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        log_path=tmp_path / "review_log.jsonl",
        input_fn=_scripted("a", "r", "q"),
        console=_quiet_console(),
    )
    verified = _read_jsonl(tmp_path / "verified.jsonl")
    log = _read_jsonl(tmp_path / "review_log.jsonl")
    assert [v["qid"] for v in verified] == ["q1"]
    assert verified[0]["verified"] is True
    assert verified[0]["question"] == "How do I declare a query parameter?"
    assert log == [{"qid": "q2", "decision": "rejected"}]
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["remaining"] == 1


def test_edit_then_accept_stores_edited_fields(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, CANDIDATES)
    review(
        candidates_path=candidates,
        verified_path=tmp_path / "verified.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        log_path=tmp_path / "review_log.jsonl",
        input_fn=_scripted("e", "Edited question?", "", "a", "q"),
        console=_quiet_console(),
    )
    verified = _read_jsonl(tmp_path / "verified.jsonl")
    assert verified[0]["question"] == "Edited question?"
    assert verified[0]["reference_answer"] == "Use Query from fastapi."


def test_flagged_items_are_not_verified(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, CANDIDATES)
    review(
        candidates_path=candidates,
        verified_path=tmp_path / "verified.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        log_path=tmp_path / "review_log.jsonl",
        input_fn=_scripted("f", "q"),
        console=_quiet_console(),
    )
    verified = _read_jsonl(tmp_path / "verified.jsonl")
    log = _read_jsonl(tmp_path / "review_log.jsonl")
    assert verified == []
    assert log == [{"qid": "q1", "decision": "flagged"}]


def test_resume_skips_already_reviewed(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, CANDIDATES)
    log_path = tmp_path / "review_log.jsonl"
    log_path.write_text(json.dumps({"qid": "q1", "decision": "rejected"}) + "\n", encoding="utf-8")

    class _SecondOnly:
        def __init__(self):
            self.calls = 0
            self.answers = ["a", "q"]

        def __call__(self, _prompt=""):
            self.calls += 1
            if self.calls > 2:
                raise AssertionError("too many prompts")
            return self.answers[self.calls - 1]

    summary = review(
        candidates_path=candidates,
        verified_path=tmp_path / "verified.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        log_path=log_path,
        input_fn=_SecondOnly(),
        console=_quiet_console(),
    )
    verified = _read_jsonl(tmp_path / "verified.jsonl")
    assert [v["qid"] for v in verified] == ["q2"]
    assert summary["accepted"] == 1
    assert summary["skipped"] == 1
    assert summary["remaining"] == 1


def test_invalid_key_reprompts(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, CANDIDATES)
    summary = review(
        candidates_path=candidates,
        verified_path=tmp_path / "verified.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        log_path=tmp_path / "review_log.jsonl",
        input_fn=_scripted("x", "a", "q"),
        console=_quiet_console(),
    )
    assert summary["accepted"] == 1


def test_excerpts_shown_when_documents_available(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, CANDIDATES[:1])
    docs = tmp_path / "documents.jsonl"
    docs.write_text(
        json.dumps(
            {
                "doc_id": "guide.md",
                "title": "Guide",
                "text": "Use Query() to declare query parameters in FastAPI.",
                "sections": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    console = _quiet_console()
    review(
        candidates_path=candidates,
        verified_path=tmp_path / "verified.jsonl",
        documents_path=docs,
        log_path=tmp_path / "review_log.jsonl",
        input_fn=_scripted("r", "q"),
        console=console,
    )
    shown = console.file.getvalue()
    assert "How do I declare a query parameter?" in shown
    assert "Use Query() to declare query parameters" in shown
