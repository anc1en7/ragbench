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

## Pending human actions

- [x] Provide `LLM_BASE_URL` + `LLM_API_KEY` — done 2026-10-28; written to local `.env` (gitignored): base `https://vps.taila5d33a.ts.net:8444/v1`.
- [x] **Restore one working chat provider on OmniRoute, then confirm the pinned `LLM_MODEL`** — resolved 2026-10-28: user connected the `rag` combo (Mistral Ministral 14B / Grok pool); `ragbench llm-probe` green, pinned in `results/hardware.json` and `.env`.
- [x] Gold-set review to >= 150 verified questions (Checkpoint 1) — **done 2026-10-29: 157 verified**, 1 rejected, 18 flagged; split run (dev 47 / test 110).
- [ ] Approve storing the OmniRoute key as a GitHub Actions secret (needed for the RAGAS generation stage in CI)
- [ ] Dashboard deployment (Hugging Face Spaces or Streamlit Community Cloud) per `docs/DEPLOY.md`
- [ ] Record demo video and capture one Phoenix trace screenshot

## Open technical questions

- ~~**OmniRoute chat routes all failing (2026-10-28)**~~ — **resolved 2026-10-29**: user updated OmniRoute (fixed the stale `x-grok-client-version` 0.2.106 → 1.0.41 that xAI's proxy rejected with 426) and connected the `rag` combo. Grok Build (`gc/*`) works but trips xAI `model_cooldown` 429s under burst concurrency; the `rag` combo is stable at ≥1.4 req/s. Generation used `rag` with a 0.5s request pacer (`--pacer`).
- ~~**FastAPI docs layout~~** — **resolved 2026-10-28**: `docs/en/docs` exists at tag `0.162.0` (33 top-level entries, 11 subdirs, 156 `.md` files); license confirmed `MIT` by the fetcher at fetch time. No adaptation needed.
- **Actions runner time budget**: if the staged sweep exceeds the practical runtime on a free CPU runner, fall back to a reduced grid or one local run (will be logged here with measured runtimes).
