"""Gold-set candidate generation with quality filters (spec 5.6)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from ragbench.goldset.filters import (
    DEDUP_THRESHOLD,
    MAX_CODE_FENCE_RATIO,
    MAX_HEADING_COPY_RATIO,
    MAX_LEXICAL_OVERLAP,
    code_fence_ratio,
    dedup,
    heading_copy_ratio,
    lexical_overlap,
)
from ragbench.goldset.prompts import (
    build_judge_prompt,
    build_question_prompt,
    parse_judge_answer,
    parse_question_json,
)
from ragbench.goldset.schema import Candidate, GoldSpan, QuestionType
from ragbench.llm_client import RateLimitError
from ragbench.types import Document, Section

MIN_SECTION_TOKENS = 80
SHORT_BUCKET_TOKENS = 300
MEDIUM_BUCKET_TOKENS = 800
GENERATION_MAX_TOKENS = 700
JUDGE_MAX_TOKENS = 8
LLM_RETRIES = 5
RETRY_BASE_SECONDS = 2.0

_SKIP_PATH_RE = re.compile(r"changelog|release[-_ ]?notes|tutorial[-_ ]?index", re.IGNORECASE)

_PACER_LOCK = threading.Lock()
_PACER_NEXT_AT = 0.0
PACER_MIN_INTERVAL = 1.5


class Pacer:
    """Thread-safe global rate pacer: spaces request starts across all workers."""

    def __init__(self, min_interval: float = PACER_MIN_INTERVAL) -> None:
        self.min_interval = min_interval

    def wait(self) -> None:
        global _PACER_NEXT_AT
        with _PACER_LOCK:
            now = time.monotonic()
            start = max(now, _PACER_NEXT_AT)
            _PACER_NEXT_AT = start + self.min_interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


_PACER = Pacer()


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...


class QuestionEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class SectionSample:
    doc_id: str
    section_id: str
    heading_path: str
    text: str
    n_tokens: int
    bucket: str
    start: int
    end: int


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"documents file not found: {path} (run corpus ingest first)")
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            sections = [Section(**section) for section in raw.pop("sections", [])]
            documents.append(Document(sections=sections, **raw))
    return documents


def _bucket_for(n_tokens: int) -> str:
    if n_tokens < SHORT_BUCKET_TOKENS:
        return "short"
    if n_tokens < MEDIUM_BUCKET_TOKENS:
        return "medium"
    return "long"


def eligible_sections(doc: Document) -> list[SectionSample]:
    samples: list[SectionSample] = []
    for section in doc.sections:
        if _SKIP_PATH_RE.search(section.heading_path) or _SKIP_PATH_RE.search(doc.doc_id):
            continue
        text = doc.text[section.start : section.end]
        n_tokens = len(text.split())
        if n_tokens < MIN_SECTION_TOKENS:
            continue
        if code_fence_ratio(text) > MAX_CODE_FENCE_RATIO:
            continue
        samples.append(
            SectionSample(
                doc_id=doc.doc_id,
                section_id=section.section_id,
                heading_path=section.heading_path,
                text=text,
                n_tokens=n_tokens,
                bucket=_bucket_for(n_tokens),
                start=section.start,
                end=section.end,
            )
        )
    return samples


def sample_sections(documents: list[Document], target: int, seed: int = 42) -> list[SectionSample]:
    """Stratified sample: round-robin over (document, length-bucket) groups."""
    groups: dict[tuple[str, str], list[SectionSample]] = {}
    for doc in documents:
        for sample in eligible_sections(doc):
            groups.setdefault((sample.doc_id, sample.bucket), []).append(sample)
    rng = random.Random(seed)
    for pool in groups.values():
        rng.shuffle(pool)
    picked: list[SectionSample] = []
    keys = sorted(groups)
    while len(picked) < target and any(groups[key] for key in keys):
        for key in keys:
            if len(picked) >= target:
                break
            if groups[key]:
                picked.append(groups[key].pop())
    return picked


def _chat_with_retry(client: ChatClient, messages: list[dict[str, str]], max_tokens: int) -> str:
    last_error: Exception | None = None
    for attempt in range(LLM_RETRIES):
        _PACER.wait()
        try:
            return client.chat(
                messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except RateLimitError as exc:
            last_error = exc
            if attempt >= LLM_RETRIES - 1:
                break
            time.sleep(min(float(exc.retry_after), 120.0) + random.uniform(0, 1.0))
        except Exception as exc:  # noqa: BLE001 - any transport/decode error is retryable
            last_error = exc
            if attempt >= LLM_RETRIES - 1:
                break
            time.sleep(RETRY_BASE_SECONDS * 2**attempt + random.uniform(0, 0.5))
    raise RuntimeError(
        f"LLM call failed after {LLM_RETRIES} attempts: {last_error}"
    ) from last_error


def _make_qid(doc_id: str, section_id: str, question: str) -> str:
    digest = hashlib.sha256(f"{doc_id}|{section_id}|{question}".encode()).hexdigest()
    return f"q{digest[:12]}"


def _generate_one(
    sample: SectionSample, client: ChatClient
) -> tuple[dict[str, str] | None, str | None]:
    """Returns (parsed, None) on success or (None, drop_reason)."""
    prompt = build_question_prompt(sample.heading_path, sample.text)
    try:
        raw = _chat_with_retry(client, [{"role": "user", "content": prompt}], GENERATION_MAX_TOKENS)
        parsed = parse_question_json(raw)
    except (ValueError, RuntimeError):
        return None, "generation_error"
    if heading_copy_ratio(parsed["question"], sample.heading_path) >= MAX_HEADING_COPY_RATIO:
        return None, "heading_copy"
    if lexical_overlap(parsed["question"], sample.text) >= MAX_LEXICAL_OVERLAP:
        return None, "lexical_overlap"
    return parsed, None


def _judge_one(
    item: dict[str, str], sample: SectionSample, client: ChatClient
) -> tuple[bool | None, str | None]:
    prompt = build_judge_prompt(item["question"], item["reference_answer"], sample.text)
    try:
        raw = _chat_with_retry(client, [{"role": "user", "content": prompt}], JUDGE_MAX_TOKENS)
        return parse_judge_answer(raw), None
    except (ValueError, RuntimeError):
        return None, "judge_error"


def _run_pool[ItemT, ResultT](
    items: list[ItemT], worker: Callable[[ItemT], ResultT], concurrency: int
) -> list[tuple[ItemT, ResultT]]:
    results: list[tuple[ItemT, ResultT]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            results.append((futures[future], future.result()))
    return results


def generate_candidates(
    documents_path: Path,
    out_path: Path,
    stats_path: Path,
    client: ChatClient,
    embedder: QuestionEmbedder,
    sample_target: int = 800,
    concurrency: int = 4,
    seed: int = 42,
    pacer_interval: float = PACER_MIN_INTERVAL,
) -> dict[str, Any]:
    """Generate filtered candidates; returns a stats dict (also written to stats_path)."""
    _PACER.min_interval = pacer_interval
    documents = load_documents(documents_path)
    samples = sample_sections(documents, target=sample_target, seed=seed)
    drops: dict[str, int] = {
        "generation_error": 0,
        "heading_copy": 0,
        "lexical_overlap": 0,
        "judge_no": 0,
        "judge_error": 0,
        "duplicate": 0,
    }

    generated: list[tuple[dict[str, str], SectionSample]] = []
    for sample, (parsed, reason) in _run_pool(
        samples, lambda s: _generate_one(s, client), concurrency
    ):
        if parsed is None:
            drops[reason or "generation_error"] += 1
        else:
            generated.append((parsed, sample))

    judged: list[tuple[dict[str, str], SectionSample]] = []
    for (parsed, sample), (supported, reason) in _run_pool(
        generated, lambda pair: _judge_one(pair[0], pair[1], client), concurrency
    ):
        if supported is None:
            drops[reason or "judge_error"] += 1
        elif not supported:
            drops["judge_no"] += 1
        else:
            judged.append((parsed, sample))

    judged.sort(key=lambda pair: pair[0]["question"])
    if judged:
        vectors = embedder.embed([item["question"] for item, _ in judged])
    else:
        vectors = np.zeros((0, 1))
    dropped = dedup(vectors, threshold=DEDUP_THRESHOLD)

    candidates: list[Candidate] = []
    for index, (parsed, sample) in enumerate(judged):
        if index in dropped:
            drops["duplicate"] += 1
            continue
        candidates.append(
            Candidate(
                qid=_make_qid(sample.doc_id, sample.section_id, parsed["question"]),
                question=parsed["question"],
                reference_answer=parsed["reference_answer"],
                gold_sections=[sample.section_id],
                gold_spans=[GoldSpan(doc_id=sample.doc_id, start=sample.start, end=sample.end)],
                question_type=cast(QuestionType, parsed["question_type"]),
                source_doc=sample.doc_id,
                verified=False,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(candidate.model_dump_json() + "\n")

    stats: dict[str, Any] = {
        "seed": seed,
        "sample_target": sample_target,
        "sampled": len(samples),
        "thresholds": {
            "heading_copy_ratio": MAX_HEADING_COPY_RATIO,
            "lexical_overlap": MAX_LEXICAL_OVERLAP,
            "dedup_cosine": DEDUP_THRESHOLD,
            "max_code_fence_ratio": MAX_CODE_FENCE_RATIO,
            "min_section_tokens": MIN_SECTION_TOKENS,
        },
        "drops": drops,
        "survivors": len(candidates),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def fastembed_embedder(model: str = "BAAI/bge-small-en-v1.5") -> QuestionEmbedder:
    """ONNX MiniLM/BGE embedder for dedup (no torch). Lazy import."""

    class _FastEmbed:
        name = model

        def __init__(self) -> None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=model)

        def embed(self, texts: list[str]) -> np.ndarray:
            vectors = list(self._model.embed(texts))
            arr = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms

    return _FastEmbed()


__all__ = [
    "SectionSample",
    "eligible_sections",
    "fastembed_embedder",
    "generate_candidates",
    "load_documents",
    "sample_sections",
]
