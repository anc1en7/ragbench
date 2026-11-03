# Open questions and deviations

Deviations from `RAGBENCH_AGENT_SPEC.md` and items awaiting human input.
Format: status, decision, reason, date.

## Decided (with the human)

| # | Item | Decision | Reason | Date |
|---|---|---|---|---|
| 1 | LLM for generation + judge | **OmniRoute on private VPS** via `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` (not Ollama) | Endpoint already available; spec section 2 explicitly allows any OpenAI-compatible endpoint via env vars | 2026-10-25 |
| 2 | Local services | **Docker Desktop on Windows** (not a bare Linux docker engine) | Host OS is Windows 11; Desktop provides the engine + WSL2 backend | 2026-10-25 |
| 3 | Sweep execution | **GitHub Actions** (`workflow_dispatch` job on free CPU runner for the public repo); regression eval on every PR | Human does not want to run heavy compute on their own PC; public repo gets free Actions minutes | 2026-10-25 |
| 4 | Gold-set review | Human reviews **all** candidates via `make goldset-review`; target >= 150 verified | Credibility core of the project | 2026-10-25 |
| 5 | Hosting | Public GitHub repo **`ragbench`**, pushed incrementally per phase | Portfolio showcase | 2026-10-25 |
| 6 | LLM model pin | **`LLM_MODEL=rag`** — user's OmniRoute combo (pools Mistral Ministral 14B / Grok 4.5). Verified: 15/15 burst probe, full 506-section generation run with zero cooldowns | Grok Build OAuth route (`gc/*`) kept hitting xAI `model_cooldown` 429s under burst; the `rag` combo has no such token-bucket | 2026-10-28 |

## Phase 3 deviations (2026-11-02)

| # | Item | Decision | Reason |
|---|---|---|---|
| P3-1 | Span-overlap hit rule | `overlap >= min(f*chunk_len, f*gold_len)` (spec's "20% of the chunk **or** 20% of the gold span"), not the plan's `max(...)` | Spec governs; `max` would reject a small chunk wholly inside a large gold section and a large chunk wholly containing a small span |
| P3-2 | Run file name | `results/runs/{config_id}__{split}.json`; `summary.csv` keyed by `(config_id, split)` | Plan's `{config_id}.json` would let a config's test run overwrite its dev run |
| P3-3 | Dataset selection | `cfg.corpus` picks the data: `fastapi_docs` (full) or `mini` (fixture). Splits for `mini` = its qids in `splits.json` (9 dev / 31 test) | Keeps `config_id` tied to the data it was evaluated on |
| P3-4 | Query embedding cache | `embed_query` no longer uses the disk cache | Cached query vectors made `embed_query` latency an `np.load` (0.6 ms) instead of inference (~38 ms) |
| P3-5 | Eval always re-indexes in-process | Every `ragbench eval` indexes before querying (document embeddings come from the cache) | Required for the in-process control_bm25; index caching is a Phase 4 sweep concern |

## Phase 4 deviations (2026-11-02)

| # | Item | Decision | Reason |
|---|---|---|---|
| P4-1 | Selection metric | Dev winners ranked by **nDCG@10**, tiebreak MRR@10 then Recall@5 (`configs/grid.yaml`), not the plan's Recall@5 | Recall@5 is saturated (1.000 on the dev baseline) and cannot rank configs |
| P4-2 | Sweep host and device | Sweep ran **locally on the RTX 3050** (human decision 2026-11-02), not on GitHub Actions. CUDA torch `2.15.0+cu132` is installed in the local `.venv` only; `pyproject.toml`, `uv.lock` and CI stay on CPU torch. Locally set `UV_NO_SYNC=1` (or use `uv run --no-sync`), because any sync reinstalls the CPU wheel. This includes `make` targets and the pre-commit hooks, which call `uv run`: a `git commit` without `UV_NO_SYNC=1` silently swapped torch back to CPU once during Phase 4 (caught before any test-split run; all 33 dev runs record `+cu132`) | CPU-only estimate was 15+ h (cross-encoder ~1-2 pairs/s); GPU brings it to about 1-2 h. All configs are measured on the same device |
| P4-3 | Dependency pins | `rich` 15.0.0 -> **14.3.4**; `langchain-community==0.4.1`; `requires-python` bounded `<3.15` | ragas 0.4.3 needs instructor (rich<15) and imports a langchain-community module removed after 0.4.1; instructor has no 3.15 release |
| P4-4 | Judge client | RAGAS judge uses `openai.AsyncOpenAI` through ragas `llm_factory` (instructor, JSON mode) on the same `LLM_BASE_URL`; the generator uses `OpenAICompatClient`. Judge model = `LLM_JUDGE_MODEL` if set, else `LLM_MODEL` (`rag`), the same model as the generator | ragas 0.4 structured output only works through instructor-patched clients. Same-model judging carries self-preference bias (LIMITATIONS) |
| P4-5 | `generation.enabled` in sweep configs | Stays `false` (retrieval sweep); RAGAS on finalists still uses each config's `context_k`/`temperature` | Flipping it would change every `config_id` and orphan the retrieval runs |
| P4-6 | Stage parents | A child stage expands the parent stage's `top_n` configs; the child's own `top_n` controls what it passes on | Makes `estimate` exact and keeps the grid readable |
| P4-7 | First (CPU) sweep attempt | The 12 CPU-measured runs and the CPU-computed embedding cache were set aside (not committed) and everything was re-run on the GPU, including the Phase 3 baseline (`3384ade0ba3e`) | Latency is only comparable when measured on one device |
| P4-8 | RAGAS run complete | All 5 finalists judged on 100 test questions, n_valid 100/100 on every metric, 0 judge errors; `docs/judge_spotcheck.md` written (20 items). Finalist 5 was interrupted once by an OmniRoute quota exhaustion (HTTP 429 `ALL_TARGETS_SKIPPED` across the whole `rag` pool); the run was stopped before anything was saved and re-run after the reset. The `rag` combo routes to different underlying models (Grok, Mistral), so the generator and judge model is not fixed across finalists: finalists `cabfa4deefe8` and `ceb37f9c707e` had identical contexts on all 100 questions yet differ by 0.06 in faithfulness | Completed 2026-11-03. Open: whether to pin a single underlying judge/generator model for a re-run, and whether to revise the `gen-v1` prompt (33-56% of answers are "context does not contain the answer" despite context recall ~0.95) |

## Pending human actions

- [x] Provide `LLM_BASE_URL` + `LLM_API_KEY` — done 2026-10-28; written to local `.env` (gitignored): base `https://vps.taila5d33a.ts.net:8444/v1`.
- [x] **Restore one working chat provider on OmniRoute, then confirm the pinned `LLM_MODEL`** — resolved 2026-10-28: user connected the `rag` combo (Mistral Ministral 14B / Grok pool); `ragbench llm-probe` green, pinned in `results/hardware.json` and `.env`.
- [x] Gold-set review to >= 150 verified questions (Checkpoint 1) — **done 2026-10-29: 157 verified**, 1 rejected, 18 flagged; split run (dev 47 / test 110).
- [x] Run the deferred RAGAS stage (P4-8) — done 2026-11-03
- [x] Checkpoint 4 accepted as-is (2026-11-03). The human chose to finish the project over improving accuracy. No pgvector FTS fix, no single-model RAGAS re-run, and no `gen-v1` prompt revision; these are recorded as known issues in `docs/LIMITATIONS.md`
- [x] (Optional) Review `docs/judge_spotcheck.md` (20 items) — skipped (optional, 2026-11-03)
- [ ] Approve storing the OmniRoute key as a GitHub Actions secret (needed for the RAGAS generation stage in CI)
- [x] Push `main` and confirm CI passes on GitHub — done 2026-11-03 (run 41605820826, all 3 jobs green)
- [x] Dashboard deployment — done 2026-11-03: <https://ragbench.streamlit.app/> (Streamlit Community Cloud; HF Docker Spaces now require PRO)
- [x] Record demo video and capture one Phoenix trace screenshot — **skipped by human decision (2026-11-03)**; storyboard kept in `docs/DEMO_STORYBOARD.md`
- [x] Fresh-clone reproduction test (separate directory, fresh `make setup`) — **skipped by human decision (2026-11-03)**; in-tree `make reproduce` + `ragbench reproduce --finalists` stand

## Phase 5-6 deviations (2026-11-03)

| # | Item | Decision | Reason |
|---|---|---|---|
| P5-1 | Regression negative evidence | `configs/regression_degraded.yaml` uses `final_k: 3`, not 16-token chunks | 16-token chunks do not regress under span-overlap scoring (Recall@5 unchanged at 0.871); a tiny chunk inside a gold section is a hit. Evidence in METHODOLOGY |
| P5-2 | Regression thresholds | Recall@5 max drop 0.03 (spec), plus MRR@10 max drop 0.05 | Closes the Phase 3 question of whether the gate should also watch MRR@10 |
| P5-3 | Baseline device | `eval/baseline.json` generated on the local GPU; CI checks on CPU | Embeddings agree to float tolerance; a near-tie flip would show as a small drop well inside 0.03 |
| P5-4 | API indexing | The API indexes a config on first query when its namespace is empty | Needed for the in-process BM25 control and for configs never indexed in the running backends |
| P5-5 | Compose profiles | `api` and `dashboard` services are in the `app` profile | `make up` stays a fast backends-only start; the API image is large |
| P6-1 | Pareto figure | Dependency-free SVG (`docs/figures/pareto.svg`) instead of PNG | No image-export dependency (kaleido/matplotlib) needed; renders on GitHub |
| P6-2 | `make reproduce` depths | fast (default) / `--finalists` / `--full` | The full sweep is about 80 min on GPU; the fast path checks the corpus lock, the gate and the README |

## Open technical questions

- ~~**OmniRoute chat routes all failing (2026-10-28)**~~ — **resolved 2026-10-29**: user updated OmniRoute (fixed the stale `x-grok-client-version` 0.2.106 → 1.0.41 that xAI's proxy rejected with 426) and connected the `rag` combo. Grok Build (`gc/*`) works but trips xAI `model_cooldown` 429s under burst concurrency; the `rag` combo is stable at ≥1.4 req/s. Generation used `rag` with a 0.5s request pacer (`--pacer`).
- ~~**FastAPI docs layout~~** — **resolved 2026-10-28**: `docs/en/docs` exists at tag `0.162.0` (33 top-level entries, 11 subdirs, 156 `.md` files); license confirmed `MIT` by the fetcher at fetch time. No adaptation needed.
- **Actions runner time budget**: if the staged sweep exceeds the practical runtime on a free CPU runner, fall back to a reduced grid or one local run (will be logged here with measured runtimes).
- ~~**Recall@5 saturation (Phase 3, hypothesis)**~~ — **resolved 2026-11-02:** dev selection uses nDCG@10 (P4-1); the CI-gate question (add MRR@10?) remains for Phase 5. Original note: the baseline dense config scores Recall@5 = Recall@10 = 1.000 on dev (47 q). Hypothesis: section-derived questions plus 512-token chunks make gold sections easy to reach within the top 5. Recall@1, MRR@10 and nDCG@10 still separate configs. Decision needed before Phase 4: choose the dev-selection metric (suggest nDCG@10 or MRR@10), and decide whether the CI regression gate should watch MRR@10 as well as Recall@5.
- ~~**Cold embedding throughput (Phase 4)**~~ — **resolved 2026-11-02:** `ragbench bench-embed` measures cache-bypassing throughput per embedder into `results/embedder_throughput.json`. Original note: chunks/sec is cache-dependent (Phase 3 baseline: 125.6 chunks/s on the first index run vs 1093.6 on a warm re-run). The sweep should record throughput from a cold-cache pass per embedder, or report it as cache-dependent.
- **pgvector sparse (FTS) is near-useless on natural-language questions (Phase 4, decision needed):** `websearch_to_tsquery` (spec section 5.3) ANDs every term, so FTS returned any hits for only 3/47 dev questions (nDCG@10 ~0.04-0.05). As a result **pgvector `hybrid` equals pgvector `dense`** in every sweep run, while the in-process BM25 control (OR semantics) scores ~0.82. Stage B therefore measured the query construction, not "lexical vs dense". Candidate fix: OR the lexemes (`to_tsquery` with `|`, or `websearch_to_tsquery` on terms joined by ` or `), keep `ts_rank_cd`, then re-run stages B-D and finalists (~30 min GPU). Qdrant's BM25 sparse path (fastembed) is unaffected. Until decided, README/findings label pgvector sparse/hybrid rows accordingly.
- **Cross-encoder latency on the laptop GPU:** bge-reranker-base runs on `cuda:0` but takes ~650 ms for 20 pairs and ~1.6 s for 50 (fp32, 512 tokens; measured directly). That is ~50x the no-rerank p95. fp16 or a smaller reranker would cut it; it is not changed so that all reported runs share one setup.
