import json
from dataclasses import asdict

import numpy as np
import pytest

from ragbench.goldset.generate import (
    eligible_sections,
    generate_candidates,
    sample_sections,
)
from ragbench.types import Document, Section

PROSE = ("prose word " * 150).strip()


def _doc(doc_id: str, sections: list[tuple[str, str, str]]) -> Document:
    """Build a Document from (slug, heading_path, body) tuples."""
    text_parts = []
    built = []
    pos = 0
    for slug, heading_path, body in sections:
        chunk = f"# {heading_path.split(' > ')[-1]}\n\n{body}\n\n"
        start = pos
        end = pos + len(chunk)
        text_parts.append(chunk)
        built.append(
            Section(section_id=f"{doc_id}#{slug}", heading_path=heading_path, start=start, end=end)
        )
        pos = end
    return Document(doc_id=doc_id, title=doc_id, text="".join(text_parts), sections=built)


def _long(body: str = PROSE) -> str:
    return body


def test_eligible_sections_skips_short_changelog_and_code():
    doc = _doc(
        "doc.md",
        [
            ("intro", "Intro", _long()),
            ("tiny", "Tiny", "only a few words"),
            ("changelog", "Changelog", _long()),
            (
                "code",
                "Examples",
                "```python\n" + "x = 1\n" * 60 + "```",
            ),
        ],
    )
    eligible = eligible_sections(doc)
    ids = {s.section_id for s in eligible}
    assert ids == {"doc.md#intro"}
    assert eligible[0].bucket == "long" or eligible[0].bucket == "medium"


def test_eligible_sections_assigns_buckets():
    short_body = ("word " * 150).strip()  # ~150 tokens -> short bucket
    doc = _doc("a.md", [("s", "S", short_body)])
    assert eligible_sections(doc)[0].bucket == "short"


def test_sample_sections_is_deterministic_and_stratified():
    docs = [
        _doc("d1.md", [("s1", "A", _long()), ("s2", "B", _long())]),
        _doc("d2.md", [("s1", "A", _long()), ("s2", "B", _long())]),
    ]
    first = sample_sections(docs, target=2, seed=42)
    second = sample_sections(docs, target=2, seed=42)
    assert [s.section_id for s in first] == [s.section_id for s in second]
    assert len(first) == 2
    assert {s.doc_id for s in first} == {"d1.md", "d2.md"}


def test_sample_sections_target_larger_than_pool_returns_all():
    docs = [_doc("d1.md", [("s1", "A", _long())])]
    picked = sample_sections(docs, target=10, seed=42)
    assert len(picked) == 1


class FakeClient:
    """Deterministic chat stub: answers generation vs judge prompts by content."""

    def __init__(self, judge_no_for: frozenset[str] = frozenset()):
        self.judge_no_for = judge_no_for
        self.calls = 0

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "Answer with exactly" in prompt:
            for marker in self.judge_no_for:
                if marker in prompt:
                    return "no"
            return "yes"
        heading = prompt.split("## Heading path\n", 1)[1].split("\n", 1)[0]
        payload = {
            "question": f"What should you know about {heading}?",
            "reference_answer": f"Everything in {heading}.",
            "question_type": "factual",
        }
        return json.dumps(payload)


class FakeEmbedder:
    def embed(self, texts):
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(texts), 8))


def _write_documents(path, docs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(asdict(doc)) + "\n")


def test_generate_candidates_end_to_end(tmp_path):
    docs = [
        _doc("d1.md", [("s1", "Alpha", _long()), ("s2", "Beta", _long())]),
        _doc("d2.md", [("s1", "Gamma", _long())]),
    ]
    documents = tmp_path / "documents.jsonl"
    _write_documents(documents, docs)
    out = tmp_path / "candidates.jsonl"
    stats = tmp_path / "generate_stats.json"

    result = generate_candidates(
        documents_path=documents,
        out_path=out,
        stats_path=stats,
        client=FakeClient(),
        embedder=FakeEmbedder(),
        sample_target=10,
        concurrency=2,
        seed=42,
        pacer_interval=0,
    )
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    assert all(item["verified"] is False for item in lines)
    assert all(item["question_type"] == "factual" for item in lines)
    qids = [item["qid"] for item in lines]
    assert len(set(qids)) == len(qids)
    assert result["survivors"] == 3
    meta = json.loads(stats.read_text(encoding="utf-8"))
    assert meta["thresholds"]["heading_copy_ratio"] == 0.6
    assert meta["survivors"] == 3


def test_generate_candidates_drops_judge_no(tmp_path):
    docs = [_doc("d1.md", [("s1", "Alpha", _long()), ("s2", "Beta", _long())])]
    documents = tmp_path / "documents.jsonl"
    _write_documents(documents, docs)
    result = generate_candidates(
        documents_path=documents,
        out_path=tmp_path / "c.jsonl",
        stats_path=tmp_path / "s.json",
        client=FakeClient(judge_no_for=frozenset({"Alpha"})),
        embedder=FakeEmbedder(),
        sample_target=10,
        concurrency=1,
        seed=42,
        pacer_interval=0,
    )
    assert result["drops"]["judge_no"] == 1
    assert result["survivors"] == 1


def test_generate_candidates_qid_is_stable(tmp_path):
    docs = [_doc("d1.md", [("s1", "Alpha", _long())])]
    documents = tmp_path / "documents.jsonl"
    _write_documents(documents, docs)

    def run(path):
        return generate_candidates(
            documents_path=documents,
            out_path=path,
            stats_path=path.parent / "s.json",
            client=FakeClient(),
            embedder=FakeEmbedder(),
            sample_target=10,
            concurrency=1,
            seed=42,
            pacer_interval=0,
        )

    run(tmp_path / "a.jsonl")
    run(tmp_path / "b.jsonl")
    a = (tmp_path / "a.jsonl").read_text(encoding="utf-8")
    b = (tmp_path / "b.jsonl").read_text(encoding="utf-8")
    assert a == b
    assert a.strip()


def test_generate_candidates_missing_documents_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_candidates(
            documents_path=tmp_path / "nope.jsonl",
            out_path=tmp_path / "c.jsonl",
            stats_path=tmp_path / "s.json",
            client=FakeClient(),
            embedder=FakeEmbedder(),
            sample_target=10,
            concurrency=1,
            seed=42,
            pacer_interval=0,
        )
