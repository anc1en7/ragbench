"""Terminal review tool for gold-set candidates (spec 5.6).

Keys: ``a``ccept, ``e``dit, ``r``eject, ``f``lag, ``q``uit. Decisions are
appended immediately so the review is resumable; accepted items land in
``verified.jsonl`` with ``verified: true``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from ragbench.goldset.schema import Candidate

EXCERPT_LIMIT = 600

DecisionKeys = {"a", "e", "r", "f", "q"}


def load_candidates(path: Path) -> list[Candidate]:
    if not path.exists():
        raise FileNotFoundError(f"candidates file not found: {path} (run generation first)")
    return [
        Candidate.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _reviewed_qids(verified_path: Path, log_path: Path) -> set[str]:
    seen = {rec["qid"] for rec in _read_jsonl(verified_path)}
    seen |= {rec["qid"] for rec in _read_jsonl(log_path)}
    return seen


def _load_documents(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(rec["doc_id"]): rec for rec in _read_jsonl(path)}


def _excerpt(doc: dict[str, Any], candidate: Candidate) -> str:
    text = str(doc.get("text", ""))
    parts: list[str] = []
    for span in candidate.gold_spans:
        if span.doc_id != doc.get("doc_id"):
            continue
        chunk = text[span.start : span.end].strip()
        if len(chunk) > EXCERPT_LIMIT:
            chunk = chunk[:EXCERPT_LIMIT] + " ..."
        if chunk:
            parts.append(chunk)
    return "\n--- gold span ---\n".join(parts)


def _show(console: Console, candidate: Candidate, documents: dict[str, dict[str, Any]]) -> None:
    console.print(
        f"[bold]{candidate.qid}[/] ({candidate.question_type}) "
        f"source_doc={candidate.source_doc} sections={candidate.gold_sections}"
    )
    console.print(f"Q: {candidate.question}", markup=False)
    console.print(f"A: {candidate.reference_answer}", markup=False)
    if candidate.gold_spans:
        doc = documents.get(candidate.gold_spans[0].doc_id)
        if doc is not None:
            excerpt = _excerpt(doc, candidate)
            if excerpt:
                console.print(excerpt, style="green", markup=False)


def _edit(candidate: Candidate, input_fn: Callable[[str], str], console: Console) -> None:
    console.print("Editing (empty line keeps current value).")
    question = input_fn(f"question [{candidate.question}]: ").strip()
    if question:
        candidate.question = question
    answer = input_fn(f"reference answer [{candidate.reference_answer}]: ").strip()
    if answer:
        candidate.reference_answer = answer


def review(
    candidates_path: Path = Path("data/goldset/candidates.jsonl"),
    verified_path: Path = Path("data/goldset/verified.jsonl"),
    documents_path: Path = Path("data/documents.jsonl"),
    log_path: Path = Path("data/goldset/review_log.jsonl"),
    *,
    input_fn: Callable[[str], str] = input,
    console: Console | None = None,
) -> dict[str, int]:
    """Run the interactive review loop; returns decision counts."""
    console = console or Console()
    candidates = load_candidates(candidates_path)
    reviewed = _reviewed_qids(verified_path, log_path)
    documents = _load_documents(documents_path)
    accepted = 0
    rejected = 0
    flagged = 0
    skipped = 0

    for candidate in candidates:
        if candidate.qid in reviewed:
            skipped += 1
            continue
        _show(console, candidate, documents)
        key = ""
        while key not in {"a", "r", "f", "q"}:
            key = input_fn("[a]ccept / [e]dit / [r]eject / [f]lag / [q]uit: ").strip().lower()
            if key == "e":
                _edit(candidate, input_fn, console)
                key = ""
            elif key not in {"a", "r", "f", "q"}:
                console.print("invalid key, try again", style="yellow")
        if key == "q":
            break
        reviewed.add(candidate.qid)
        if key == "a":
            payload = candidate.model_copy(update={"verified": True})
            _append_jsonl(verified_path, payload.model_dump())
            accepted += 1
        else:
            decision = "rejected" if key == "r" else "flagged"
            _append_jsonl(log_path, {"qid": candidate.qid, "decision": decision})
            if decision == "rejected":
                rejected += 1
            else:
                flagged += 1

    return {
        "accepted": accepted,
        "rejected": rejected,
        "flagged": flagged,
        "skipped": skipped,
        "remaining": len(candidates) - len(reviewed),
    }
