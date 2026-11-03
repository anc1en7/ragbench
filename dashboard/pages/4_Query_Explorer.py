"""Query explorer: one gold question, two configs, side by side (Task 5.4)."""

from __future__ import annotations

import html

import streamlit as st
from data import documents, gold_questions, label, runs

st.set_page_config(page_title="Query explorer · RAGBench", layout="wide")
st.title("Query explorer")

split = st.selectbox("Split", ["test", "dev"])
available = sorted(runs(split), key=lambda r: -r.metrics["ndcg@10"].mean)
if len(available) < 1:
    st.info("No runs for this split.")
    st.stop()
gold = gold_questions()
qids = [q.qid for q in available[0].per_query]
qid = st.selectbox(
    "Question",
    qids,
    format_func=lambda q: (
        f"{gold[q]['question']}  ({gold[q]['question_type']})" if q in gold else q
    ),
)
question = gold.get(qid, {})
with st.expander("Reference answer and gold spans"):
    st.write(question.get("reference_answer", ""))
    st.write([f"{s['doc_id']} [{s['start']}:{s['end']}]" for s in question.get("gold_spans", [])])

labels = {r.config_id: f"{label(r)}  ({r.config_id})" for r in available}
picks = st.columns(2)
left = picks[0].selectbox("Config A", list(labels), format_func=labels.get, index=0)
right = picks[1].selectbox(
    "Config B", list(labels), format_func=labels.get, index=min(1, len(labels) - 1)
)

docs = documents()
if docs is None:
    st.caption(
        "Chunk text is shown when the corpus is ingested locally (`make ingest`); "
        "otherwise only document offsets are shown."
    )
gold_spans = question.get("gold_spans", [])


def _highlight(doc_id: str, start: int, end: int) -> str:
    """Chunk text with the overlap with any gold span highlighted."""
    assert docs is not None
    text = docs.get(doc_id, "")[start:end]
    marks = sorted(
        (max(s["start"], start) - start, min(s["end"], end) - start)
        for s in gold_spans
        if s["doc_id"] == doc_id and s["start"] < end and s["end"] > start
    )
    out, pos = [], 0
    for a, b in marks:
        if a < pos:
            a = pos
        out.append(html.escape(text[pos:a]))
        out.append(f"<mark>{html.escape(text[a:b])}</mark>")
        pos = max(pos, b)
    out.append(html.escape(text[pos:]))
    return "".join(out)[:1200]


for col, config_id in zip(st.columns(2), (left, right), strict=True):
    result = next(r for r in available if r.config_id == config_id)
    query = next(q for q in result.per_query if q.qid == qid)
    with col:
        st.markdown(f"**{label(result)}**")
        st.caption(
            " · ".join(f"{k} {v:.3f}" for k, v in query.metrics.items())
            + f" · run p95 {result.latency_ms['total'].p95:.0f} ms"
        )
        for rank, (hit, relevant) in enumerate(
            zip(query.hits, query.relevance, strict=True), start=1
        ):
            badge = "✅ gold hit" if relevant else "· miss"
            with st.container(border=True):
                st.markdown(
                    f"**#{rank}** {badge} · score {hit.score:.3f} · `{hit.doc_id}` "
                    f"[{hit.start}:{hit.end}]"
                )
                if docs is not None:
                    st.markdown(_highlight(hit.doc_id, hit.start, hit.end), unsafe_allow_html=True)
