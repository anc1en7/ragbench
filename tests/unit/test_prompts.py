import json

import pytest

from ragbench.goldset.prompts import (
    build_judge_prompt,
    build_question_prompt,
    parse_judge_answer,
    parse_question_json,
)


def test_question_prompt_contains_heading_and_section():
    prompt = build_question_prompt(heading_path="Guide > Install", section_text="Run pip install.")
    assert "Guide > Install" in prompt
    assert "Run pip install." in prompt


def test_judge_prompt_contains_question_answer_and_section():
    prompt = build_judge_prompt(
        question="What is X?", reference_answer="X is Y.", section_text="X is Y indeed."
    )
    assert "What is X?" in prompt
    assert "X is Y." in prompt
    assert "X is Y indeed." in prompt


def test_parse_question_json_plain():
    payload = {
        "question": "What is FastAPI?",
        "reference_answer": "A web framework.",
        "question_type": "factual",
    }
    parsed = parse_question_json(json.dumps(payload))
    assert parsed["question"] == "What is FastAPI?"
    assert parsed["question_type"] == "factual"


def test_parse_question_json_wrapped_in_fences():
    payload = {"question": "Q?", "reference_answer": "A.", "question_type": "how_to"}
    text = f"```json\n{json.dumps(payload)}\n```"
    parsed = parse_question_json(text)
    assert parsed["question"] == "Q?"


def test_parse_question_json_rejects_bad_type():
    payload = {"question": "Q?", "reference_answer": "A.", "question_type": "essay"}
    with pytest.raises(ValueError, match="question_type"):
        parse_question_json(json.dumps(payload))


def test_parse_question_json_rejects_empty_fields():
    payload = {"question": "  ", "reference_answer": "A.", "question_type": "factual"}
    with pytest.raises(ValueError):
        parse_question_json(json.dumps(payload))


def test_parse_question_json_rejects_non_json():
    with pytest.raises(ValueError):
        parse_question_json("I cannot answer that.")


def test_parse_judge_answer_yes_variants():
    assert parse_judge_answer("yes") is True
    assert parse_judge_answer("Yes, the answer is supported.") is True


def test_parse_judge_answer_no_variants():
    assert parse_judge_answer("no") is False
    assert parse_judge_answer("No — the section does not mention it.") is False


def test_parse_judge_answer_garbage_raises():
    with pytest.raises(ValueError):
        parse_judge_answer("perhaps")
