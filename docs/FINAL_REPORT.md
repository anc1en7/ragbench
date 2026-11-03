# RAGBench final report

Date: 2026-11-03. Covers Phases 0-6 of `RAGBENCH_AGENT_SPEC.md`.

## What was built

- **Corpus and gold set.** The FastAPI docs, pinned to tag `0.162.0` (156 files, 2,199
  sections). 157 human-verified questions with span-level labels, split into dev 47 and
  test 110.
- **Pipelines.** Three chunkers (fixed, recursive_md, semantic), three embedders
  (bge-small-en-v1.5, e5-small-v2, all-MiniLM-L6-v2), dense / sparse / hybrid-RRF retrieval
  plus an in-process BM25 control, no reranker or a cross-encoder (bge-reranker-base), and
  two backends (pgvector, Qdrant).
- **Evaluation.** Chunker-agnostic span-overlap relevance; Recall@{1,3,5,10}, MRR@10 and
  nDCG@10 with 95% bootstrap CIs; per-stage p50/p95 latency; indexing time, size and RSS; an
  ANN-vs-exact sanity check; RAGAS on the finalists.
- **Sweep.** A staged sweep: 33 configs on dev, then 5 dev-selected finalists on test, with
  generated findings and paired-bootstrap ablations.
- **Serving.** A FastAPI service (`/health`, `/configs`, `/query`, `/eval/summary`, OpenAPI at
  `/docs`) with an LRU pipeline cache. OpenTelemetry stage tracing to Phoenix behind
  `TRACING=1`. Docker images and compose services for the API and the dashboard.
- **CI.** Lint + unit tests, a regression eval on the mini fixture with pgvector and Qdrant
  service containers (fails if Recall@5 drops more than 0.03), and Docker builds.
- **Dashboard.** Streamlit in replay mode with five pages: leaderboard, Pareto, ablations,
  query explorer and methodology. It needs no torch.
- **Docs.** `make report` regenerates the README results, findings and Pareto SVG.
  `make reproduce` runs the clean-clone check. Methodology, limitations, architecture,
  deploy guide, demo storyboard and resume drafts are all in `docs/`.

## Headline results (from `results/`)

- Best finalist on test: recursive_md-256 / bge-small-en-v1.5 / Qdrant hybrid /
  cross-encoder@20. nDCG@10 0.887 [0.839, 0.931], Recall@5 0.927 [0.873, 0.973], p95
  1231.8 ms. All 5 finalists are statistically tied.
- Cross-encoder reranking: +0.057 nDCG@10 on dev [+0.009, +0.108] at +844 ms p95.
  Backend: no detectable difference (0/3 pairs).
- RAGAS: context recall 0.947-0.960, but faithfulness 0.463-0.699 and answer relevancy
  0.290-0.421, because 32-55% of answers are "context does not contain the answer"
  refusals.

## Commands

```bash
make setup                      # install (CPU torch) + pre-commit hooks
cp .env.example .env            # set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
make up                         # pgvector + Qdrant
make ingest                     # fetch pinned corpus, build data/documents.jsonl
ragbench eval --config configs/baseline_dense.yaml --split dev
ragbench sweep --grid configs/grid.yaml          # staged sweep on dev (resumable)
ragbench finalize --grid configs/grid.yaml       # finalists -> test -> RAGAS -> findings
make report                     # regenerate README results + docs/figures/pareto.svg
make reproduce                  # clean-clone check (fast path)
ragbench reproduce --finalists  # also re-evaluate the 5 finalists and compare
make regression-check           # CI gate locally
make api                        # http://localhost:8000/docs
make dashboard                  # http://localhost:8501
docker compose --profile app up -d                # API + dashboard in Docker
docker compose --profile phoenix up -d            # Phoenix trace viewer (TRACING=1)
```

## Versions and hardware

- Python 3.13. Package pins are in `pyproject.toml` / `uv.lock`; key versions, model
  revisions and the Qdrant/pgvector images are in `docs/METHODOLOGY.md` (Resolved versions).
- Machine: Intel i5-12450H (12 threads), RTX 3050 Laptop GPU (4 GB), 16 GB RAM, Windows 11,
  Docker Desktop. Details in `results/hardware.json`. All benchmark runs used CUDA torch
  `2.15.0+cu132`. CI and Docker images use CPU torch.
- LLM: OmniRoute combo `rag` (router pool) for gold-set generation, answer generation and
  RAGAS judging.

## Compute time

- Staged sweep (33 dev runs, GPU): about 81 minutes wall clock (first to last run timestamp).
- Finalists on test (5 runs) plus RAGAS: the RAGAS stage spans about 21 hours of wall clock
  because the run was paused overnight and once waited for an LLM quota reset. Active
  compute time was not recorded separately.
- Regression gate on the mini fixture: under 1 minute on this machine once models are
  cached.

## Known issues

1. pgvector sparse search uses `websearch_to_tsquery` (AND semantics). It returns hits for
   only 3/47 dev questions, so pgvector hybrid equals pgvector dense. Not fixed (Checkpoint 4
   decision).
2. The `rag` LLM is a router pool. RAGAS answer metrics vary by about ±0.1 between runs, and
   the same model generates and judges (self-preference risk).
3. High refusal rate from prompt `gen-v1` (32-55%). Not revised.
4. Recall@5 is saturated on dev, so selection uses nDCG@10.
5. The API Docker image is large (about 5.4 GB) because the package depends on torch, RAGAS
   and LangChain. A slimmer API-only dependency set would cut this.

## Deviations from the spec

All are logged with reasons in `docs/OPEN_QUESTIONS.md`. The main ones:

- The sweep ran locally on the GPU, not on GitHub Actions (P4-2).
- Dev selection uses nDCG@10, not Recall@5 (P4-1).
- The regression gate's negative evidence uses `final_k: 3`, not 16-token chunks. Under
  span-overlap scoring, 16-token chunks do not regress (`docs/METHODOLOGY.md`, Regression
  gate).
- The README Pareto figure is a dependency-free SVG instead of a PNG.
- Checkpoint 4 was accepted as-is, without the FTS fix or a single-model RAGAS re-run
  (human decision, 2026-11-03).

## Reproduction check (2026-11-03)

- `make reproduce` (fast path) passed. The corpus matched `corpus.lock`
  (156 documents, 2,199 sections), the mini regression gate reproduced the baseline
  exactly (Recall@5 0.8710, MRR@10 0.6863), and the README blocks regenerated identically.
- `ragbench reproduce --finalists` passed (**REPRODUCED**). All 5 test finalists were
  re-evaluated from scratch on the same GPU in about 35 minutes. All 30 retrieval metrics
  (6 per finalist) matched the committed runs **exactly** (difference 0.0000; tolerance
  0.01).
- Not done: a run in a separate fresh clone directory with a fresh `make setup`. The checks
  above ran in the working tree with the existing environment. RAGAS was not re-run (it
  needs the LLM endpoint, and its scores vary by about ±0.1 between runs).

## Remaining human actions

- [x] Dashboard deployed on Streamlit Community Cloud: <https://ragbench.streamlit.app/> (2026-11-03).
- [x] Pushed to GitHub; CI passed on `1b305e5` (lint-test, regression-eval, docker-build).
- [ ] Record the demo (`docs/DEMO_STORYBOARD.md`) and save one Phoenix trace screenshot as
      `docs/phoenix_trace.png`.
- [ ] Optional: review the 20 RAGAS items in `docs/judge_spotcheck.md`.
- [x] GitHub release [v0.1.0](https://github.com/anc1en7/ragbench/releases/tag/v0.1.0) with all run JSONs, summary.csv, gold set and Pareto figure.
- [ ] Optional: approve storing the OmniRoute key as an Actions secret (only needed to run
      RAGAS in CI).
