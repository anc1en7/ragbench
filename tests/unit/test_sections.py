import json

from ragbench.corpus.build import build_documents
from ragbench.corpus.clean import clean_markdown
from ragbench.corpus.sections import parse_sections
from ragbench.types import Document, Section


def test_clean_strips_front_matter_and_normalizes():
    raw = (
        "---\ninclude_yaml:\n  topic_repos: data/topic_repos.yml\n---\n"
        "# Title\r\n\r\n\r\n\r\n\r\nBody\r\n"
    )
    out = clean_markdown(raw)
    assert out.startswith("# Title")
    assert "include_yaml" not in out
    assert "\r" not in out
    assert "Body" in out


def test_clean_collapses_blank_runs_outside_fences_only():
    raw = "# T\n\n\n\n\nbody\n\n```python\n# comment\n\n\n\ncode\n```\nend\n"
    out = clean_markdown(raw)
    assert "# T\n\nbody" in out
    assert "# comment\n\n\n\ncode" in out
    assert out.endswith("end\n")


def test_sections_offsets_paths_and_ids():
    text = "# Title\n\nintro\n\n## Alpha\n\nalpha body\n\n### A1\n\na1 body\n\n## Beta\n\nbeta body"
    title, sections = parse_sections("doc.md", text)
    assert title == "Title"
    assert len(sections) == 4
    slices = [text[s.start : s.end] for s in sections]
    assert slices == [
        "# Title\n\nintro\n\n",
        "## Alpha\n\nalpha body\n\n",
        "### A1\n\na1 body\n\n",
        "## Beta\n\nbeta body",
    ]
    assert [s.heading_path for s in sections] == [
        "Title",
        "Title > Alpha",
        "Title > Alpha > A1",
        "Title > Beta",
    ]
    assert [s.section_id for s in sections] == [
        "doc.md#title",
        "doc.md#alpha",
        "doc.md#a1",
        "doc.md#beta",
    ]
    for s in sections:
        assert 0 <= s.start < s.end <= len(text)


def test_heading_anchor_becomes_slug_and_title_strips_anchor():
    text = "# Alternatives { #alternatives-inspiration-and-comparisons }\n\nbody"
    title, sections = parse_sections("alternatives.md", text)
    assert title == "Alternatives"
    assert sections[0].section_id == "alternatives.md#alternatives-inspiration-and-comparisons"
    assert sections[0].heading_path == "Alternatives"


def test_headings_inside_code_fences_are_ignored():
    text = "# Real\n\n```python\n# not a heading\nprint(1)\n```\n\nafter\n"
    _title, sections = parse_sections("doc.md", text)
    assert len(sections) == 1
    assert "# not a heading" in text[sections[0].start : sections[0].end]


def test_duplicate_slugs_are_disambiguated():
    text = "# A\n\n## Example\n\nx\n\n## Example\n\ny"
    _title, sections = parse_sections("doc.md", text)
    ids = [s.section_id for s in sections]
    assert ids == ["doc.md#a", "doc.md#example", "doc.md#example-2"]


def test_offsets_survive_cleaning():
    raw = "---\nx: 1\n---\n# T\n\n\n\n\nbody\n\n## S\n\nmore"
    text = clean_markdown(raw)
    _title, sections = parse_sections("doc.md", text)
    assert sections[0].start == 0
    assert text[sections[0].start : sections[0].end] == "# T\n\nbody\n\n"
    assert text[sections[1].start : sections[1].end] == "## S\n\nmore"


def test_document_without_headings_has_no_sections():
    title, sections = parse_sections("links.md", "just a list\n\n- a\n- b\n")
    assert title == ""
    assert sections == []


def test_build_documents_writes_jsonl(tmp_path):
    raw = tmp_path / "raw"
    (raw / "guide").mkdir(parents=True)
    (raw / "index.md").write_text("# FastAPI\n\nHello\n\n## Features\n\nfast\n", encoding="utf-8")
    (raw / "guide" / "setup.md").write_text(
        "---\nx: 1\n---\n# Setup\n\nInstall it\n", encoding="utf-8"
    )
    out = tmp_path / "documents.jsonl"
    docs = build_documents(raw_dir=raw, out=out)
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [d["doc_id"] for d in lines] == ["guide/setup.md", "index.md"]
    assert lines[1]["title"] == "FastAPI"
    first = lines[1]["sections"][0]
    assert set(first) == {"section_id", "heading_path", "start", "end"}
    assert lines[0]["sections"][0]["section_id"] == "guide/setup.md#setup"
    doc = docs[1]
    assert isinstance(doc.sections[0], Section)
    assert doc.text[doc.sections[1].start : doc.sections[1].end] == "## Features\n\nfast\n"
