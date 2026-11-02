"""RAGAS generation metrics for finalist configs (Task 4.3, spec section 5.7).

API verified against the installed ``ragas==0.4.3`` source (the ``ragas.metrics.collections``
"modern" metrics): each metric is constructed with an instructor-backed LLM from
``ragas.llms.base.llm_factory(model, client=AsyncOpenAI(...))`` and scored with an async
``ascore(...)``:

- ``Faithfulness(llm).ascore(user_input, response, retrieved_contexts)``
- ``AnswerRelevancy(llm, embeddings, strictness=3).ascore(user_input, response)``
- ``ContextPrecision(llm).ascore(user_input, reference, retrieved_contexts)`` (reference-based)
- ``ContextRecall(llm).ascore(user_input, retrieved_contexts, reference)``

The judge is reached through the same OpenAI-compatible endpoint as generation
(``LLM_BASE_URL``/``LLM_API_KEY``) using the official ``openai.AsyncOpenAI`` client,
because ragas 0.4 drives structured output through ``instructor``, which patches that
client; ``OpenAICompatClient`` is still the generator. The judge model is
``LLM_JUDGE_MODEL`` when set, else ``LLM_MODEL``. When the two are the same model,
self-preference bias applies (docs/LIMITATIONS.md). Judge temperature is 0. No ``seed`` is
sent: the ``rag`` combo routes to upstream models that reject it as an unknown parameter
(every such call failed), so determinism relies on temperature 0 only. ``top_p`` is set to
1.0 because ragas defaults it to 0.1, which Mistral rejects with greedy (temperature 0)
sampling; at temperature 0 ``top_p`` does not change the output.

AnswerRelevancy embeds questions with the local ``BAAI/bge-small-en-v1.5`` (query side,
via :class:`RagbenchEmbeddings`) so no embedding API is needed. Metric failures
(provider errors, unparsable judge output) and NaN scores are recorded as ``None`` and
excluded from that metric's mean; ``n_valid`` reports how many questions each mean
covers.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ragas.embeddings.base import BaseRagasEmbedding

from ragbench.eval.latency import tokens_per_query
from ragbench.eval.results import (
    GENERATION_METRICS,
    GenerationMetrics,
    GenerationQuery,
    MetricCI,
    TokenStats,
)
from ragbench.eval.stats import bootstrap_ci
from ragbench.generation.prompt import PROMPT_VERSION, SYSTEM_PROMPT, ChatClient, generate_answer
from ragbench.goldset.schema import Candidate
from ragbench.llm_client import LLMSettings
from ragbench.pipeline import Pipeline
from ragbench.types import Embedder

SUBSET_SIZE: int = 100
ANSWER_RELEVANCY_STRICTNESS: int = 3
JUDGE_TEMPERATURE: float = 0.0
JUDGE_MAX_TOKENS: int = 2048
RAGAS_EMBEDDER: str = "BAAI/bge-small-en-v1.5"
ENV_JUDGE_MODEL = "LLM_JUDGE_MODEL"


class RagbenchEmbeddings(BaseRagasEmbedding):
    """ragas embedding adapter over a ragbench :class:`Embedder` (query-side vectors)."""

    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        self._embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return [float(x) for x in self._embedder.embed_query(text)]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


@dataclass
class GenSample:
    qid: str
    question: str
    reference: str
    contexts: list[str]
    context_ids: list[str]
    answer: str
    tokens: int


class Scorer(Protocol):
    async def score(self, sample: GenSample) -> dict[str, float | None]: ...


def _clean(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


class RagasScorer:
    """Scores one sample on the four spec metrics; a failing metric yields ``None``."""

    def __init__(self, metrics: Mapping[str, Any]) -> None:
        """``metrics`` maps each name in ``GENERATION_METRICS`` to an object with an async
        ``ascore(**kwargs)`` (ragas metric classes have per-metric keyword signatures)."""
        missing = set(GENERATION_METRICS) - set(metrics)
        if missing:
            raise ValueError(f"RagasScorer needs metrics {sorted(missing)}")
        self.metrics = dict(metrics)
        self.errors: dict[str, int] = dict.fromkeys(GENERATION_METRICS, 0)
        self.last_error: dict[str, str] = {}

    def _kwargs(self, name: str, s: GenSample) -> dict[str, Any]:
        if name == "faithfulness":
            return {
                "user_input": s.question,
                "response": s.answer,
                "retrieved_contexts": s.contexts,
            }
        if name == "answer_relevancy":
            return {"user_input": s.question, "response": s.answer}
        return {
            "user_input": s.question,
            "reference": s.reference,
            "retrieved_contexts": s.contexts,
        }

    async def _one(self, name: str, sample: GenSample) -> float | None:
        try:
            result = await self.metrics[name].ascore(**self._kwargs(name, sample))
        except Exception as exc:  # noqa: BLE001 - judge/provider failures become missing
            self.errors[name] += 1
            self.last_error[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return None
        return _clean(getattr(result, "value", result))

    async def score(self, sample: GenSample) -> dict[str, float | None]:
        values = await asyncio.gather(*(self._one(n, sample) for n in GENERATION_METRICS))
        return dict(zip(GENERATION_METRICS, values, strict=True))


def judge_model_name(settings: LLMSettings) -> str:
    return os.environ.get(ENV_JUDGE_MODEL) or settings.model


def build_ragas_scorer(settings: LLMSettings, embedder: Embedder) -> RagasScorer:
    """Real ragas 0.4.3 metrics against the configured OpenAI-compatible endpoint."""
    from openai import AsyncOpenAI
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key, max_retries=4)
    llm = llm_factory(
        judge_model_name(settings),
        client=client,
        temperature=JUDGE_TEMPERATURE,
        top_p=1.0,  # ragas defaults to 0.1; Mistral rejects top_p != 1 with temperature 0
        max_tokens=JUDGE_MAX_TOKENS,
    )
    embeddings = RagbenchEmbeddings(embedder)
    return RagasScorer(
        {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(
                llm=llm, embeddings=embeddings, strictness=ANSWER_RELEVANCY_STRICTNESS
            ),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
    )


def select_subset(
    questions: Sequence[Candidate], n: int = SUBSET_SIZE, seed: int = 42
) -> list[Candidate]:
    """Fixed seeded subset of ``n`` questions (all if fewer), returned qid-sorted."""
    ordered = sorted(questions, key=lambda q: q.qid)
    if len(ordered) <= n:
        return ordered
    picked = random.Random(seed).sample(ordered, n)
    return sorted(picked, key=lambda q: q.qid)


async def _score_all(
    scorer: Scorer, samples: Sequence[GenSample], concurrency: int
) -> list[dict[str, float | None]]:
    gate = asyncio.Semaphore(concurrency)

    async def run(sample: GenSample) -> dict[str, float | None]:
        async with gate:
            return await scorer.score(sample)

    return list(await asyncio.gather(*(run(s) for s in samples)))


def run_generation_eval(
    pipeline: Pipeline,
    questions: Sequence[Candidate],
    *,
    generator: ChatClient,
    scorer: Scorer,
    generator_model: str,
    judge_model: str,
    subset_seed: int,
    concurrency: int = 3,
    n_resamples: int = 1000,
    log: Callable[[str], None] = lambda _msg: None,
) -> tuple[GenerationMetrics, list[GenSample]]:
    """Retrieve, generate and judge every question; aggregate means with bootstrap CIs.

    Returns the metrics block plus the samples (with context texts, for the spot-check).
    """
    if not questions:
        raise ValueError("run_generation_eval needs at least one question")
    cfg = pipeline.cfg
    context_k = cfg.generation.context_k

    # retrieval is serialized (one pipeline/connection); generation calls run in a pool
    retrieved: list[tuple[Candidate, list[Any]]] = []
    for question in questions:
        hits, _ = pipeline.retrieve(question.question)
        retrieved.append((question, hits[:context_k]))

    def generate(item: tuple[Candidate, list[Any]]) -> GenSample:
        question, top = item
        contexts = [hit.chunk.text for hit in top]
        answer = generate_answer(
            generator, question.question, contexts, temperature=cfg.generation.temperature
        )
        return GenSample(
            qid=question.qid,
            question=question.question,
            reference=question.reference_answer,
            contexts=contexts,
            context_ids=[hit.chunk.chunk_id for hit in top],
            answer=answer,
            tokens=tokens_per_query(SYSTEM_PROMPT + "\n" + question.question, contexts),
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(generate, retrieved))
    log(f"generated {len(samples)} answers; judging with {judge_model}")
    scores = asyncio.run(_score_all(scorer, samples, concurrency))
    errors = getattr(scorer, "errors", None)
    if errors is not None:
        log(f"judge errors per metric (recorded as missing): {dict(errors)}")

    metrics: dict[str, MetricCI] = {}
    n_valid: dict[str, int] = {}
    for name in GENERATION_METRICS:
        values = [s[name] for s in scores if s[name] is not None]
        n_valid[name] = len(values)
        if values:
            floats = [float(v) for v in values if v is not None]
            lo, hi = bootstrap_ci(floats, n_resamples, subset_seed)
            metrics[name] = MetricCI(mean=float(np.mean(floats)), ci_lo=lo, ci_hi=hi)

    tokens = np.asarray([s.tokens for s in samples], dtype=np.float64)
    block = GenerationMetrics(
        generator_model=generator_model,
        judge_model=judge_model,
        prompt_version=PROMPT_VERSION,
        ragas_version=metadata.version("ragas"),
        ragas_embedder=RAGAS_EMBEDDER,
        answer_relevancy_strictness=ANSWER_RELEVANCY_STRICTNESS,
        temperature=cfg.generation.temperature,
        judge_temperature=JUDGE_TEMPERATURE,
        context_k=context_k,
        subset_seed=subset_seed,
        n_questions=len(samples),
        metrics=metrics,
        n_valid=n_valid,
        tokens_per_query=TokenStats(
            mean=float(tokens.mean()),
            p50=float(np.percentile(tokens, 50)),
            p95=float(np.percentile(tokens, 95)),
        ),
        per_query=[
            GenerationQuery(
                qid=s.qid,
                answer=s.answer,
                context_chunk_ids=s.context_ids,
                tokens=s.tokens,
                scores=score,
            )
            for s, score in zip(samples, scores, strict=True)
        ],
        created_at=datetime.now(UTC).isoformat(),
    )
    return block, samples


def write_spotcheck(
    path: Path,
    items: Sequence[tuple[str, GenerationQuery, Candidate, list[str]]],
    *,
    judge_model: str,
    n: int = 20,
    seed: int = 42,
) -> Path:
    """Write ``n`` seeded judged items for the human audit (spec section 5.7).

    ``items`` are ``(config_id, generation query, gold question, context texts)``.
    """
    pool = sorted(items, key=lambda it: (it[0], it[2].qid))
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    lines = [
        "# Judge spot-check",
        "",
        f"{len(picked)} RAGAS-judged items sampled with seed {seed} from the finalist configs "
        f"(judge model `{judge_model}`). The human checks each verdict and ticks one box; "
        "the agent does not fill these in.",
        "",
    ]
    for idx, (config_id, gq, question, contexts) in enumerate(picked, start=1):
        scores = gq.scores
        lines += [
            f"## {idx}. `{question.qid}` (config `{config_id}`)",
            "",
            f"**Question:** {question.question}",
            "",
            f"**Reference answer:** {question.reference_answer}",
            "",
            f"**Generated answer:** {gq.answer}",
            "",
            "**Retrieved contexts:**",
            "",
        ]
        for c_idx, text in enumerate(contexts, start=1):
            snippet = " ".join(text.split())
            lines.append(f"{c_idx}. {snippet[:500]}{'...' if len(snippet) > 500 else ''}")
        lines += ["", "| metric | judge score |", "|---|---|"]
        for name in GENERATION_METRICS:
            value = scores.get(name)
            lines.append(f"| {name} | {'n/a' if value is None else f'{value:.3f}'} |")
        lines += [
            "",
            "- [ ] Judge verdicts look right",
            "- [ ] Judge verdicts look wrong (note which metric and why):",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
