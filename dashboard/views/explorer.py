"""Query explorer: one gold question, two pipelines, side by side."""

from __future__ import annotations

import html

import streamlit as st
import ui
from data import documents, gold_questions, ranked, retrieval_label

ui.setup(
    "Query explorer",
    "Pick a gold question and compare what two pipelines retrieved for it. A hit means the "
    "chunk overlaps the human-labelled answer span.",
)

c1, c2 = st.columns([1, 3], gap="small")
split = c1.segmented_control("Split", ["test", "dev"], default="test") or "test"
available = ranked(split)
gold = gold_questions()
qids = [q.qid for q in available[0].per_query]
qid = c2.selectbox(
    "Question",
    qids,
    format_func=lambda q: gold[q]["question"] if q in gold else q,
)
question = gold.get(qid, {})
gold_docs = ", ".join(sorted({s["doc_id"] for s in question.get("gold_spans", [])}))
st.markdown(
    f'<span class="rb-chip">{html.escape(question.get("question_type", ""))}</span>'
    f'<span class="rb-chip">gold: {html.escape(gold_docs)}</span>',
    unsafe_allow_html=True,
)
with st.expander("Reference answer"):
    st.write(question.get("reference_answer", ""))

names = {
    r.config_id: retrieval_label(r.config) + f"  ({r.config.config_id[:6]})" for r in available
}
p1, p2 = st.columns(2, gap="medium")
left = p1.selectbox("Pipeline A", list(names), format_func=names.get, index=0)
right = p2.selectbox("Pipeline B", list(names), format_func=names.get, index=min(1, len(names) - 1))

docs = documents()
spans = question.get("gold_spans", [])


def highlight(doc_id: str, start: int, end: int) -> str:
    assert docs is not None
    text = docs.get(doc_id, "")[start:end]
    marks = sorted(
        (max(s["start"], start) - start, min(s["end"], end) - start)
        for s in spans
        if s["doc_id"] == doc_id and s["start"] < end and s["end"] > start
    )
    out, pos = [], 0
    for a, b in marks:
        a = max(a, pos)
        out.append(html.escape(text[pos:a]))
        out.append(f"<mark>{html.escape(text[a:b])}</mark>")
        pos = max(pos, b)
    out.append(html.escape(text[pos:]))
    return "".join(out)[:1400]


for col, config_id in zip((p1, p2), (left, right), strict=True):
    result = next(r for r in available if r.config_id == config_id)
    query = next(q for q in result.per_query if q.qid == qid)
    with col:
        chips = "".join(
            f'<span class="rb-chip">{k} {v:.2f}</span>'
            for k, v in query.metrics.items()
            if k in ("ndcg@10", "mrr@10", "recall@5")
        )
        st.markdown(chips, unsafe_allow_html=True)
        rows = []
        for rank, (hit, relevant) in enumerate(zip(query.hits, query.relevance, strict=True), 1):
            badge = (
                '<span class="rb-gold">✓ hit</span>'
                if relevant
                else '<span class="rb-miss">· miss</span>'
            )
            rows.append(
                f'<div class="rb-hit"><span class="rb-rank">{rank}</span>{badge} '
                f'<span class="rb-doc">{html.escape(hit.doc_id)}</span>'
                f'<span class="rb-score">{hit.score:.3f}</span></div>'
            )
        st.markdown(f'<div class="rb-card">{"".join(rows)}</div>', unsafe_allow_html=True)
        if docs is not None:
            with st.expander("Chunk text (gold span highlighted)"):
                for rank, hit in enumerate(query.hits[:5], 1):
                    st.markdown(f"**#{rank}** `{hit.doc_id}`")
                    st.markdown(highlight(hit.doc_id, hit.start, hit.end), unsafe_allow_html=True)

if docs is None:
    st.caption(
        "Chunk text is not bundled with the hosted dashboard (the corpus is fetched at build "
        "time). Run it locally after `make ingest` to read the retrieved passages."
    )
