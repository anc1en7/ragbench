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

## Pending human actions

- [x] Provide `LLM_BASE_URL` + `LLM_API_KEY` — done 2026-10-28; written to local `.env` (gitignored): base `https://vps.taila5d33a.ts.net:8444/v1`.
- [ ] **Restore one working chat provider on OmniRoute, then confirm the pinned `LLM_MODEL`** — `GET /v1/models` works (366 models) but every tested chat route fails (see technical questions); `ragbench llm-probe` exits 1 until one route succeeds. Blocks gold-set generation (Phase 1, Task 1.3).
- [ ] Gold-set review to >= 150 verified questions (Checkpoint 1)
- [ ] Approve storing the OmniRoute key as a GitHub Actions secret (needed for the RAGAS generation stage in CI)
- [ ] Dashboard deployment (Hugging Face Spaces or Streamlit Community Cloud) per `docs/DEPLOY.md`
- [ ] Record demo video and capture one Phoenix trace screenshot

## Open technical questions

- **OmniRoute chat routes all failing (2026-10-28):** with a valid key, `GET /v1/models` returns 366 models, but `POST /v1/chat/completions` fails on every route class tested: `oc/*` OpenCode free models → 429 free-usage-limit (or 401 missing upstream key for paid `oc/*`), `ddgw/*` DuckDuckGo → 418 anti-abuse challenge (VPS egress IP rate-limited), `felo/*` → 400 model unavailable, `tllm/*` → 403 Vercel block on server egress IP, `aug/*` → 502 stream early EOF, `pepper/*` → 502, all `auto/*` alias pools → exhausted after 5–14 attempts. **Human to connect/repair a provider in the OmniRoute dashboard (Providers), then rerun `uv run ragbench llm-probe` and set the confirmed `LLM_MODEL` in `.env`.** Until then `LLM_MODEL=oc/claude-haiku-4-5` in `.env` is provisional and unverified.
- **FastAPI docs layout**: confirm `docs/en/docs` still matches the pinned release tag at fetch time; if not, adapt and log here (spec section 2).
- **Actions runner time budget**: if the staged sweep exceeds the practical runtime on a free CPU runner, fall back to a reduced grid or one local run (will be logged here with measured runtimes).
