# Demo storyboard (about 3 minutes)

A shot list for a screen recording. Prepare first: `make up`, `docker compose --profile
phoenix up -d`, set `TRACING=1` in `.env`, run `make api` in one terminal and
`make dashboard` in another. Use a large terminal font.

| # | Time | Screen | Say |
|---|---|---|---|
| 1 | 0:00-0:15 | README top, scroll to Findings | "RAG changes are usually judged by eye. RAGBench measures them on a human-verified gold set, with confidence intervals." |
| 2 | 0:15-0:35 | `data/goldset/verified.jsonl` in the editor, then the `goldset review` tool | "157 questions, each reviewed by hand. Labels are spans in the source sections, not chunks, so different chunkers are compared fairly." |
| 3 | 0:35-1:05 | Dashboard → Pareto | "33 configs on the dev split. Colour is retrieval mode. The frontier shows what you get for each extra millisecond. The cross-encoder buys about 0.06 nDCG for about 840 ms on this laptop GPU." |
| 4 | 1:05-1:25 | Dashboard → Ablations, rerank then backend | "Every claim is a paired bootstrap. Reranking helps; the backend does not matter: 0 of 3 pgvector-vs-Qdrant pairs differ." |
| 5 | 1:25-1:45 | Dashboard → Leaderboard (test) | "Only 5 dev-selected finalists ever touched the test split. On test they are statistically tied." |
| 6 | 1:45-2:05 | Dashboard → Query explorer, a question where the configs disagree | "Here is one question across two configs, with the gold span highlighted." |
| 7 | 2:05-2:30 | Browser `http://localhost:8000/docs`, run `POST /query` | "Any config is servable through the API. The response has the contexts, per-stage timings and a trace id." |
| 8 | 2:30-2:45 | Phoenix `http://localhost:6006`, open that trace | "The trace shows each stage: embed, dense, sparse, fusion, rerank." Capture this screen as `docs/phoenix_trace.png`. |
| 9 | 2:45-3:00 | GitHub Actions run, then `docs/METHODOLOGY.md` regression section | "CI fails a pull request if Recall@5 drops more than 3 points. Here is the gate failing on a deliberately degraded config. Limitations are written down too." |

Keep it honest: mention that RAGAS answer metrics are noisy because the LLM is a router
pool, and that generation, not retrieval, is the weak part.
