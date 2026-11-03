# Resume and LinkedIn drafts

Every number below is copied from `results/summary.csv` or `results/runs/*.json` (dev
ablations use the paired bootstrap from `results/findings_draft.md`). The spec template
said "hybrid + rerank improved Recall@5"; the data does not support that wording
(Recall@5 is saturated on dev, and pgvector hybrid equals dense), so the bullets use the
effect that was actually measured.

## Resume bullets

- Built **RAGBench**, a config-driven RAG evaluation platform comparing **33** retrieval
  pipelines (3 chunkers, 3 embedders, dense/BM25/hybrid-RRF, cross-encoder reranking) across
  pgvector and Qdrant on a **157-question** human-verified gold set, with 95% bootstrap
  confidence intervals on every metric.
- Found that cross-encoder reranking raised nDCG@10 by **+0.057** (95% paired-bootstrap CI
  [+0.009, +0.108]) at **+844 ms** p95 latency, while the vector backend made no detectable
  difference (0/3 matched pairs). The best pipeline reached nDCG@10 **0.887** [0.839, 0.931]
  and Recall@5 **0.927** on a held-out 110-question test split.
- Shipped a FastAPI query service, Docker Compose stack, OpenTelemetry tracing to Phoenix,
  a Streamlit replay dashboard, and a GitHub Actions regression gate that fails a pull
  request when Recall@5 drops more than 3 points.

## LinkedIn post outline

1. **Problem:** RAG changes (chunk size, embedder, reranker, vector DB) are usually judged by
   eye on a handful of queries.
2. **Method:** 157 human-verified questions with span-level labels in the source sections,
   so chunkers are compared fairly; dev for selection, test for reporting; confidence
   intervals on every number.
3. **Surprising finding:** retrieval was not the bottleneck. RAGAS context recall was about
   0.95, but the generator answered "the context does not contain the answer" in 32-55% of
   replies. And pgvector vs Qdrant made no measurable quality difference.
4. **Links:** repo and dashboard (add the URL after deploying, see `docs/DEPLOY.md`).
