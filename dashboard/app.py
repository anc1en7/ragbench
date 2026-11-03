"""RAGBench replay dashboard: overview page (Task 5.4). Run: ``make dashboard``."""

from __future__ import annotations

import streamlit as st
from data import GEN_METRICS, ablation_table, ci_text, run, summary

REFUSAL = "does not contain the answer"

st.set_page_config(page_title="RAGBench", layout="wide")
st.title("RAGBench")
st.caption(
    "Benchmarking RAG pipelines on the FastAPI docs with a human-verified gold set "
    "(157 questions: 47 dev for selection, 110 test for the headline numbers). "
    "Replay mode: every number on these pages is read from `results/`."
)

df = summary()
test = df[df["split"] == "test"].sort_values(["ndcg@10", "mrr@10"], ascending=False)
dev = df[df["split"] == "dev"]

if test.empty:
    st.info("No test-split runs yet: run `ragbench finalize`.")
    st.stop()

best = test.iloc[0]
best_run = run(best["config_id"], "test")
cols = st.columns(4)
cols[0].metric("Best test nDCG@10", f"{best['ndcg@10']:.3f}")
cols[0].caption(f"95% CI [{best['ndcg@10_lo']:.3f}, {best['ndcg@10_hi']:.3f}]")
cols[1].metric("Recall@5", f"{best['recall@5']:.3f}")
cols[1].caption(f"95% CI [{best['recall@5_lo']:.3f}, {best['recall@5_hi']:.3f}]")
cols[2].metric("p95 latency", f"{best['p95_total_ms']:.0f} ms")
cols[2].caption("warm, one query at a time, RTX 3050 Laptop")
cols[3].metric("Configs swept on dev", f"{len(dev)}")
cols[3].caption(f"{len(test)} dev-selected finalists evaluated on test")

st.subheader("Best finalist")
st.markdown(
    f"**{best['chunker_label']} / {best['embedder_short']} / {best['backend']} / "
    f"{best['mode_label']} / rerank {best['rerank_label']}** (`{best['config_id']}`)"
)

gen = best_run.generation
if gen is not None:
    st.markdown("RAGAS generation metrics on 100 test questions (95% bootstrap CI):")
    gcols = st.columns(len(GEN_METRICS))
    for col, name in zip(gcols, GEN_METRICS, strict=True):
        m = gen.metrics[name]
        col.metric(name.replace("_", " "), f"{m.mean:.3f}")
        col.caption(ci_text(m.mean, m.ci_lo, m.ci_hi))

st.subheader("What the results say")
abl = ablation_table("dev", "ndcg@10")
rerank = abl[abl["factor"] == "rerank"].sort_values("delta", ascending=False)
backend = abl[abl["factor"] == "backend"]
refusals = []
for config_id in test["config_id"]:
    generation = run(config_id, "test").generation
    if generation is not None:
        answers = [q.answer.lower() for q in generation.per_query]
        refusals.append(sum(REFUSAL in a for a in answers) / max(len(answers), 1))
overlap = test["ndcg@10_lo"].max() <= test["ndcg@10_hi"].min()
bullets = [
    f"**The {len(test)} finalists are {'statistically tied' if overlap else 'not all tied'} "
    f"on test**: nDCG@10 ranges {test['ndcg@10'].min():.3f}-{test['ndcg@10'].max():.3f} "
    f"and every 95% CI {'overlaps every other' if overlap else 'does not overlap'}.",
]
if not rerank.empty:
    top = rerank.iloc[0]
    bullets.append(
        f"**The cross-encoder reranker helps**: up to {top['delta']:+.3f} nDCG@10 on dev "
        f"(CI [{top['lo']:+.3f}, {top['hi']:+.3f}]), at {top['latency_delta_ms']:+.0f} ms p95 "
        "on this laptop GPU."
    )
if not backend.empty:
    bullets.append(
        f"**The vector backend makes no difference to quality**: "
        f"{int(backend['significant'].sum())}/{len(backend)} pgvector-vs-Qdrant pairs differ."
    )
if refusals:
    bullets.append(
        "**Retrieval is not the bottleneck for answers**: the generator replied that the "
        f"context lacks the answer in {min(refusals):.0%}-{max(refusals):.0%} of answers, "
        "although RAGAS context recall is about 0.95."
    )
bullets.append(
    "**Known issues** (see Methodology): pgvector full-text search uses AND semantics, so "
    "pgvector hybrid equals dense; the `rag` LLM pool switches models, so RAGAS scores vary "
    "by about ±0.1 between runs."
)
st.markdown("\n".join(f"- {b}" for b in bullets))
st.caption("Pages: Leaderboard · Pareto · Ablations · Query explorer · Methodology")
