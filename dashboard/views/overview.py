"""Overview: the headline result, the finalists, and four findings."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import ui
from data import (
    GEN_METRICS,
    ablation_table,
    chunker_label,
    embedder_label,
    ranked,
    refusal_rate,
    retrieval_label,
    runs,
)

ui.setup(
    "RAGBench",
    "Which RAG retrieval pipeline actually works best? 33 pipelines on the FastAPI docs, "
    "scored against 157 human-verified questions, with 95% confidence intervals on every "
    "number. Finalists were picked on a dev split and reported on a held-out test split.",
)

test = ranked("test")
dev = runs("dev")
if not test:
    st.info("No test-split runs yet: run `ragbench finalize`.")
    st.stop()

best = test[0]
m = best.metrics
n_test = best.goldset.n_questions

# --- headline -------------------------------------------------------------------------
left, right = st.columns([1.25, 2], gap="medium")
with left:
    ui.tile(
        "Best nDCG@10 on the test split",
        f"{m['ndcg@10'].mean:.3f}",
        f"95% CI {m['ndcg@10'].ci_lo:.3f}–{m['ndcg@10'].ci_hi:.3f}<br>"
        f"{retrieval_label(best.config)}",
        hero=True,
    )
with right:
    a, b, c = st.columns(3, gap="small")
    with a:
        ui.tile(
            "Recall@5",
            f"{m['recall@5'].mean:.3f}",
            f"CI {m['recall@5'].ci_lo:.3f}–{m['recall@5'].ci_hi:.3f}",
        )
    with b:
        ui.tile(
            "p95 latency",
            f"{best.latency_ms['total'].p95 / 1000:.2f} s",
            "warm queries, laptop GPU",
        )
    with c:
        ui.tile(
            "Pipelines compared", f"{len(dev)}", f"{len(test)} finalists · {n_test} test questions"
        )
    st.markdown(
        f'<div class="rb-note" style="margin-top:12px">Every finalist uses '
        f"{chunker_label(best.config)} chunks and {embedder_label(best.config)} embeddings; "
        "they differ in backend, retrieval mode and rerank depth.</div>",
        unsafe_allow_html=True,
    )

# --- finalists ------------------------------------------------------------------------
lo_all = max(r.metrics["ndcg@10"].ci_lo for r in test)
hi_all = min(r.metrics["ndcg@10"].ci_hi for r in test)
tied = lo_all <= hi_all
st.subheader(
    f"The {len(test)} finalists are statistically tied" if tied else "Finalists on the test split"
)
labels = [retrieval_label(r.config) for r in test][::-1]
means = [r.metrics["ndcg@10"].mean for r in test][::-1]
los = [r.metrics["ndcg@10"].ci_lo for r in test][::-1]
his = [r.metrics["ndcg@10"].ci_hi for r in test][::-1]
fig = go.Figure(
    go.Scatter(
        x=means,
        y=labels,
        mode="markers+text",
        marker={"color": ui.BLUE, "size": 11, "line": {"color": ui.SURFACE, "width": 2}},
        text=[f"{v:.3f}" for v in means],
        textposition="top center",
        textfont={"color": ui.INK_2, "size": 11},
        error_x={
            "type": "data",
            "symmetric": False,
            "array": [h - v for h, v in zip(his, means, strict=True)],
            "arrayminus": [v - lo for v, lo in zip(means, los, strict=True)],
            "color": ui.BLUE,
            "thickness": 2,
            "width": 0,
        },
        hovertemplate="%{y}<br>nDCG@10 %{x:.3f}<extra></extra>",
    )
)
fig.update_xaxes(
    title="nDCG@10 on the test split, with 95% bootstrap CI",
    range=[min(los) - 0.02, max(his) + 0.02],
)
st.plotly_chart(ui.style(fig, 300, legend=False), theme=None, config={"displayModeBar": False})
if tied:
    st.caption(
        f"Every interval contains {lo_all:.3f}–{hi_all:.3f}, so the ranking among finalists "
        "is within noise. Pick on latency or simplicity instead."
    )

# --- findings -------------------------------------------------------------------------
st.subheader("What the sweep found")
abl = ablation_table("dev", "ndcg@10")


def top(factor: str) -> dict[str, object] | None:
    part = abl[abl["factor"] == factor]
    if part.empty:
        return None
    return dict(part.sort_values("delta", ascending=False).iloc[0])


cards = []
rr = top("rerank")
if rr:
    cards.append(
        (
            f"+{rr['delta']:.3f}",
            "nDCG@10 from a cross-encoder reranker",
            f"95% CI +{rr['lo']:.3f} to +{rr['hi']:.3f} on dev, at "
            f"+{rr['latency_delta_ms']:.0f} ms p95 on this laptop GPU. The clearest win, "
            "and the most expensive one.",
        )
    )
be = abl[abl["factor"] == "backend"]
if not be.empty:
    cards.append(
        (
            f"{int(be['significant'].sum())} of {len(be)}",
            "pgvector vs Qdrant pairs that differ",
            "With identical chunks, embeddings and HNSW settings, the vector database made no "
            "detectable difference to quality (paired bootstrap on dev nDCG@10).",
        )
    )
em = top("embedder")
if em:
    cards.append(
        (
            f"+{em['delta']:.3f}",
            "nDCG@10 from the better embedder",
            f"{em['change']}, with {em['held fixed']} held fixed "
            f"(95% CI +{em['lo']:.3f} to +{em['hi']:.3f}).",
        )
    )
rates = [x for x in (refusal_rate(r) for r in test) if x is not None]
recalls = [
    r.generation.metrics["context_recall"].mean
    for r in test
    if r.generation and "context_recall" in r.generation.metrics
]
if rates and recalls:
    cards.append(
        (
            f"{min(rates):.0%}–{max(rates):.0%}",
            "answers that say the context lacks the answer",
            f"Yet RAGAS context recall is {min(recalls):.2f}–{max(recalls):.2f}. Retrieval "
            "finds the right passages; the generator prompt is the weak link.",
        )
    )
for row in range(0, len(cards), 2):
    cols = st.columns(2, gap="medium")
    for col, card in zip(cols, cards[row : row + 2], strict=False):
        with col:
            ui.finding(*card)
    st.write("")

# --- answer quality -------------------------------------------------------------------
gen = best.generation
if gen is not None:
    st.subheader("Answer quality of the best finalist (RAGAS)")
    cols = st.columns(len(GEN_METRICS), gap="small")
    for col, name in zip(cols, GEN_METRICS, strict=True):
        g = gen.metrics[name]
        with col:
            ui.tile(
                name.replace("_", " ").capitalize(),
                f"{g.mean:.2f}",
                f"CI {g.ci_lo:.2f}–{g.ci_hi:.2f}",
            )
    st.caption(
        f"{gen.n_questions} test questions, judged by the `{gen.judge_model}` LLM pool. "
        "Scores vary by about ±0.1 between runs because the pool switches models. "
        "Known issues are listed under Methodology → Limitations."
    )
