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

- [ ] Provide `LLM_BASE_URL` + `LLM_API_KEY` (OmniRoute VPS) and confirm the pinned `LLM_MODEL`
- [ ] Gold-set review to >= 150 verified questions (Checkpoint 1)
- [ ] Approve storing the OmniRoute key as a GitHub Actions secret (needed for the RAGAS generation stage in CI)
- [ ] Dashboard deployment (Hugging Face Spaces or Streamlit Community Cloud) per `docs/DEPLOY.md`
- [ ] Record demo video and capture one Phoenix trace screenshot

## Open technical questions

- **FastAPI docs layout**: confirm `docs/en/docs` still matches the pinned release tag at fetch time; if not, adapt and log here (spec section 2).
- **Actions runner time budget**: if the staged sweep exceeds the practical runtime on a free CPU runner, fall back to a reduced grid or one local run (will be logged here with measured runtimes).
