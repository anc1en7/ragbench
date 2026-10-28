import numpy as np

from ragbench.goldset.filters import (
    MAX_HEADING_COPY_RATIO,
    MAX_LEXICAL_OVERLAP,
    code_fence_ratio,
    dedup,
    heading_copy_ratio,
    lexical_overlap,
)


def test_heading_copy_ratio_near_copy_is_high():
    ratio = heading_copy_ratio("What are Path Operation Decorators?", "Path Operation Decorators")
    assert ratio >= MAX_HEADING_COPY_RATIO


def test_heading_copy_ratio_paraphrased_question_is_low():
    ratio = heading_copy_ratio("How do I use Query Parameters?", "Query Parameters")
    assert ratio < MAX_HEADING_COPY_RATIO


def test_heading_copy_ratio_empty_heading_is_zero():
    assert heading_copy_ratio("What is anything?", "") == 0.0


def test_lexical_overlap_verbatim_phrase_is_high():
    section = "Use Query from fastapi to declare query parameters. Query validates the data."
    question = "Use Query from fastapi to declare query parameters"
    assert lexical_overlap(question, section) >= MAX_LEXICAL_OVERLAP


def test_lexical_overlap_paraphrase_is_low():
    section = "Use Query from fastapi to declare query parameters. Query validates the data."
    question = "Which object do I import to declare query parameters?"
    assert lexical_overlap(question, section) < MAX_LEXICAL_OVERLAP


def test_lexical_overlap_empty_question_is_zero():
    assert lexical_overlap("", "some text") == 0.0


def test_dedup_drops_near_duplicates_keeps_first():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.999, 0.02, 0.0])  # cosine sim ~1.0 with a
    c = np.array([0.0, 1.0, 0.0])
    dropped = dedup(np.stack([a, b, c]), threshold=0.92)
    assert dropped == {1}


def test_dedup_keeps_distinct_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert dedup(np.stack([a, b]), threshold=0.92) == set()


def test_dedup_empty_input():
    assert dedup(np.zeros((0, 4))) == set()


def test_dedup_handles_unnormalized_vectors():
    a = np.array([3.0, 0.0])
    b = np.array([5.0, 0.1])
    dropped = dedup(np.stack([a, b]), threshold=0.92)
    assert dropped == {1}


def test_code_fence_ratio_mostly_code_is_high():
    text = "# Title\n\n```python\ndef f():\n    pass\n```\n\n```python\nx = 1\ny = 2\n```"
    assert code_fence_ratio(text) > 0.6


def test_code_fence_ratio_prose_is_low():
    text = (
        "# Title\n\nSome prose here with words.\n\nMore prose lines here.\n\n"
        "And even more prose.\n\n```py\nx = 1\n```"
    )
    assert code_fence_ratio(text) < 0.5


def test_code_fence_ratio_unclosed_fence_does_not_crash():
    assert code_fence_ratio("text\n```python\nx = 1\n") >= 0.0
