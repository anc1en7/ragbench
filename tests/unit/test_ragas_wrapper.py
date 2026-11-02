"""Unit tests for the RAGAS runner wrapper (Task 4.3) — offline, no LLM calls.

Fakes stand in for the generator chat client, the ragas metrics and the pipeline. The
real ragas metric classes are only constructed (never called) to pin the 0.4.3 API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragbench.config import PipelineConfig
from ragbench.eval.ragas_runner import (
    GenSample,
    RagasScorer,
    RagbenchEmbeddings,
    build_ragas_scorer,
    run_generation_eval,
    select_subset,
    write_spotcheck,
)
from ragbench.eval.results import GENERATION_METRICS
from ragbench.generation import PROMPT_VERSION, build_messages
from ragbench.goldset.schema import Candidate, GoldSpan
from ragbench.llm_client import LLMSettings
from ragbench.pipeline import Pipeline
from ragbench.rerank import RerankNone
from ragbench.types import Chunk, Document, Hit


def _q(i: int) -> Candidate:
    return Candidate(
        qid=f"q{i:03d}",
        question=f"question {i}?",
        reference_answer=f"reference {i}",
        gold_spans=[GoldSpan(doc_id="a.md", start=0, end=10)],
        question_type="factual",
        source_doc="a.md",
        verified=True,
    )


class FakeEmbedder:
    name = "fake"
    dim = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("not used")

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeStore:
    def reset(self, namespace: str, dim: int) -> None: ...

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]:
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id="a.md", start=0, end=5, text=f"ctx {i}")
            for i in range(min(k, 8))
        ]
        return [Hit(c, 1.0 - i / 10, "dense") for i, c in enumerate(chunks)]

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]:
        return []

    def size_bytes(self, namespace: str) -> int:
        return 0


class NullChunker:
    def chunk(self, doc: Document) -> list[Chunk]:
        return []


class FakeChat:
    def __init__(self) -> None:
        self.n = 0

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):  # type: ignore[no-untyped-def]
        self.n += 1
        return "an answer"


class FakeScorer:
    """Scripted scores; question q001 gets a judge failure on context_recall."""

    async def score(self, sample: GenSample) -> dict[str, float | None]:
        out: dict[str, float | None] = dict.fromkeys(GENERATION_METRICS, 0.5)
        out["faithfulness"] = 1.0 if sample.qid == "q000" else 0.0
        if sample.qid == "q001":
            out["context_recall"] = None
        assert len(sample.contexts) == 5  # context_k from config
        return out


def _pipeline() -> Pipeline:
    cfg = PipelineConfig.model_validate(
        {
            "corpus": "mini",
            "chunker": {"type": "fixed", "size_tokens": 64, "overlap_tokens": 8},
            "embedder": {"name": "fake"},
            "backend": "pgvector",
            "retrieval": {"mode": "dense"},
            "rerank": {"type": "none"},
            "generation": {"enabled": False, "context_k": 5},
        }
    )
    return Pipeline(
        cfg,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        reranker=RerankNone(),
        chunker=NullChunker(),
    )


def test_prompt_numbers_contexts_and_is_versioned():
    messages = build_messages("How?", ["first passage", "second passage"])
    assert messages[0]["role"] == "system" and "ONLY" in messages[0]["content"]
    assert "[1] first passage" in messages[1]["content"]
    assert "[2] second passage" in messages[1]["content"]
    assert messages[1]["content"].rstrip().endswith("Answer:")
    assert PROMPT_VERSION == "gen-v1"


def test_select_subset_is_seeded_sorted_and_capped():
    questions = [_q(i) for i in range(30)]
    a = select_subset(questions, n=10, seed=42)
    assert a == select_subset(list(reversed(questions)), n=10, seed=42)  # input order irrelevant
    assert len(a) == 10 and [q.qid for q in a] == sorted(q.qid for q in a)
    assert select_subset(questions, n=10, seed=7) != a
    assert len(select_subset(questions[:4], n=10)) == 4


def test_ragbench_embeddings_use_query_side_sync_and_async():
    import asyncio

    emb = FakeEmbedder()
    adapter = RagbenchEmbeddings(emb)
    assert adapter.embed_text("x") == [1.0, 0.0, 0.0]
    assert asyncio.run(adapter.aembed_text("y")) == [1.0, 0.0, 0.0]
    assert emb.queries == ["x", "y"]


class _ScriptedMetric:
    def __init__(self, value: Any = None, exc: Exception | None = None) -> None:
        self.value, self.exc = value, exc
        self.kwargs: dict[str, Any] = {}

    async def ascore(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc:
            raise self.exc

        class R:
            value = self.value

        return R()


def test_scorer_routes_kwargs_and_turns_failures_and_nan_into_none():
    import asyncio

    metrics = {
        "faithfulness": _ScriptedMetric(0.75),
        "answer_relevancy": _ScriptedMetric(float("nan")),
        "context_precision": _ScriptedMetric(exc=RuntimeError("judge returned junk")),
        "context_recall": _ScriptedMetric(1.0),
    }
    scorer = RagasScorer(metrics)
    sample = GenSample("q1", "Q?", "ref", ["c1", "c2"], ["id1", "id2"], "ans", 10)

    scores = asyncio.run(scorer.score(sample))

    assert scores == {
        "faithfulness": 0.75,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": 1.0,
    }
    assert scorer.errors["context_precision"] == 1
    assert metrics["faithfulness"].kwargs == {
        "user_input": "Q?",
        "response": "ans",
        "retrieved_contexts": ["c1", "c2"],
    }
    assert metrics["answer_relevancy"].kwargs == {"user_input": "Q?", "response": "ans"}
    assert metrics["context_recall"].kwargs == {
        "user_input": "Q?",
        "reference": "ref",
        "retrieved_contexts": ["c1", "c2"],
    }


def test_scorer_requires_all_four_metrics():
    with pytest.raises(ValueError, match="context_recall"):
        RagasScorer({n: _ScriptedMetric(1.0) for n in GENERATION_METRICS[:3]})


def test_build_ragas_scorer_constructs_real_ragas_043_metrics_offline():
    scorer = build_ragas_scorer(
        LLMSettings(base_url="http://127.0.0.1:9/v1", api_key="x", model="judge-model"),
        FakeEmbedder(),
        seed=42,
    )
    names = {n: type(m).__name__ for n, m in scorer.metrics.items()}
    assert names == {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "AnswerRelevancy",
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
    }
    assert scorer.metrics["answer_relevancy"].strictness == 3


def test_run_generation_eval_aggregates_with_missing_values():
    chat = FakeChat()
    questions = [_q(i) for i in range(4)]

    gen, samples = run_generation_eval(
        _pipeline(),
        questions,
        generator=chat,
        scorer=FakeScorer(),
        generator_model="gen",
        judge_model="judge",
        subset_seed=42,
        n_resamples=100,
    )

    assert chat.n == 4
    assert gen.n_questions == 4 and gen.context_k == 5
    assert gen.generator_model == "gen" and gen.judge_model == "judge"
    assert gen.prompt_version == PROMPT_VERSION and gen.ragas_version == "0.4.3"
    # faithfulness: q000 = 1.0, others 0.0 -> mean 0.25 over 4 valid
    assert math.isclose(gen.metrics["faithfulness"].mean, 0.25)
    # context_recall: q001 failed -> 3 valid values of 0.5
    assert gen.n_valid["context_recall"] == 3
    assert math.isclose(gen.metrics["context_recall"].mean, 0.5)
    q001 = next(q for q in gen.per_query if q.qid == "q001")
    assert q001.scores["context_recall"] is None
    assert q001.context_chunk_ids == ["c0", "c1", "c2", "c3", "c4"]
    assert gen.tokens_per_query.mean > 0
    assert [s.qid for s in samples] == [q.qid for q in questions]
    assert samples[0].contexts[0] == "ctx 0"


def test_write_spotcheck_samples_twenty_unchecked_items(tmp_path: Path):
    from ragbench.eval.results import GenerationQuery

    items = [
        (
            f"cfg{i % 3}",
            GenerationQuery(
                qid=f"q{i:03d}",
                answer=f"answer {i}",
                context_chunk_ids=["c"],
                tokens=5,
                scores={"faithfulness": 0.5, "context_recall": None},
            ),
            _q(i),
            ["some context text"],
        )
        for i in range(40)
    ]
    path = write_spotcheck(tmp_path / "judge_spotcheck.md", items, judge_model="rag")
    text = path.read_text(encoding="utf-8")
    assert text.count("\n## ") == 20
    assert text.count("- [ ] Judge verdicts look right") == 20
    assert "- [x]" not in text
    assert "| context_recall | n/a |" in text
    assert path.read_text(encoding="utf-8") == write_spotcheck(
        tmp_path / "again.md", items, judge_model="rag"
    ).read_text(encoding="utf-8")  # deterministic
