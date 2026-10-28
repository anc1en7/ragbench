"""Shared gold-set candidate schema (spec 5.6), used by generate, review, split."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["factual", "how_to", "why_when"]
QUESTION_TYPES: tuple[str, ...] = ("factual", "how_to", "why_when")


class GoldSpan(BaseModel):
    doc_id: str
    start: int
    end: int


class Candidate(BaseModel):
    qid: str
    question: str
    reference_answer: str
    gold_sections: list[str] = Field(default_factory=list)
    gold_spans: list[GoldSpan] = Field(default_factory=list)
    question_type: QuestionType
    source_doc: str
    verified: bool = False
