"""LLM prompts and response parsing for gold-set generation (spec 5.6)."""

from __future__ import annotations

import json
import re

from ragbench.goldset.schema import QUESTION_TYPES, QuestionType

_QUESTION_MARKER = "## Heading path"
_SECTION_MARKER = "## Section text"
_JUDGE_YESNO_MARKER = 'Answer with exactly "yes" or "no".'

_QUESTION_INSTRUCTIONS = """You are writing evaluation questions for a documentation corpus.

The question must be answerable ONLY from the section text, must not copy the heading,
and must not reuse the section's phrasing verbatim.

Question types:
- factual: lookup of a specific fact
- how_to: a procedural question
- why_when: explanation of why something works or when to use it

Respond with ONLY a JSON object:
{"question": "...", "reference_answer": "...", "question_type": "factual" | "how_to" | "why_when"}
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def build_question_prompt(heading_path: str, section_text: str) -> str:
    return (
        f"{_QUESTION_INSTRUCTIONS}\n"
        f"{_QUESTION_MARKER}\n{heading_path}\n\n"
        f"{_SECTION_MARKER}\n{section_text}\n"
    )


def build_judge_prompt(question: str, reference_answer: str, section_text: str) -> str:
    return (
        "You are verifying that a reference answer is supported by a document section.\n\n"
        f"{_SECTION_MARKER}\n{section_text}\n\n"
        f"## Question\n{question}\n\n"
        f"## Reference answer\n{reference_answer}\n\n"
        f"{_JUDGE_YESNO_MARKER}\n"
    )


def parse_question_json(text: str) -> dict[str, str]:
    """Parse the generation response into {question, reference_answer, question_type}."""
    stripped = text.strip()
    fenced = _FENCE_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:120]!r}")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in response: {exc}") from exc
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("reference_answer", "")).strip()
    qtype = str(payload.get("question_type", "")).strip()
    if not question or not answer:
        raise ValueError("question/reference_answer missing or empty")
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {QUESTION_TYPES}, got {qtype!r}")
    return {"question": question, "reference_answer": answer, "question_type": qtype}


def parse_judge_answer(text: str) -> bool:
    """Parse the judge response: True = supported, False = not supported."""
    stripped = text.strip().lower()
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    raise ValueError(f"judge response is not yes/no: {text[:80]!r}")


def is_judge_prompt(prompt: str) -> bool:
    return _JUDGE_YESNO_MARKER in prompt


__all__ = [
    "QuestionType",
    "build_judge_prompt",
    "build_question_prompt",
    "is_judge_prompt",
    "parse_judge_answer",
    "parse_question_json",
]
