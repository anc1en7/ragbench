# Deploying the RAGBench dashboard

The dashboard runs in **replay mode**: it reads only committed files (`results/`,
`data/goldset/verified.jsonl`, `docs/`). It needs no database, no model and no LLM key, so
it deploys with **no secrets**. Pick one of the two hosts below. The login and deploy steps
are yours to do (spec section 5.12).

## What the deploy uses

| Path | Why |
|---|---|
| `dashboard/app.py` + `dashboard/views/` + `dashboard/ui.py` | the Streamlit app (top navigation, one file per page) |
| `dashboard/requirements.txt` | light pinned deps (streamlit, plotly, pandas, pydantic, pyyaml, numpy), **no torch** |
| `src/ragbench/` | the dashboard imports the result schemas and bootstrap stats from here (added to `sys.path` automatically, no install) |
| `results/summary.csv`, `results/runs/*.json` | every number shown |
| `data/goldset/verified.jsonl` | question text in the query explorer |
| `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md` | the Methodology page |
| `.streamlit/config.toml` | pins the light theme |

The query explorer shows chunk **text** only when `data/documents.jsonl` exists. That file is
gitignored, so a deployed dashboard shows document offsets and gold-hit flags instead of
chunk text. This is expected.

## Option A: Streamlit Community Cloud (simplest)

1. Push `main` to GitHub (`git push origin main`). The repo must be public, or you must
   connect Streamlit to your GitHub account with private-repo access.
2. Go to <https://share.streamlit.io>, sign in with GitHub, then **Create app → Deploy a
   public app from GitHub**.
3. Fill in:
   - Repository: `anc1en7/ragbench`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
   - Advanced settings → Python version: **3.13**
   - Requirements: Streamlit Cloud uses `dashboard/requirements.txt` because it sits next
     to the main file. Leave secrets empty.
4. Click **Deploy** and wait for the build (about 2-3 minutes).

## Option B: Hugging Face Spaces (Docker)

> **Needs Hugging Face PRO.** As of 2026-11-03, Hugging Face only hosts Docker Spaces for PRO
> accounts (creating one returns HTTP 402 otherwise). With PRO, `python scripts/deploy_hf_space.py
> --space <user>/ragbench` stages the needed files and uploads them in one step; it was tested
> locally. The manual steps below do the same thing.

1. Create a Space at <https://huggingface.co/new-space>: SDK **Docker**, template
   **Blank**, hardware **CPU basic (free)**.
2. Push this repo to the Space, using the Space as a second git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-user>/ragbench
   git push space main
   ```

3. Spaces needs `README.md` front matter and a root `Dockerfile`. Do this on a separate
   branch that you push only to the Space, so the GitHub README stays clean:

   ```bash
   git switch -c hf-space
   cp Dockerfile.dashboard Dockerfile
   # Prepend to README.md:
   # ---
   # title: RAGBench
   # sdk: docker
   # app_port: 8501
   # ---
   git commit -am "hf space config" && git push space hf-space:main
   ```

4. The Space builds `Dockerfile` and serves on port 8501.

## Verification checklist

- [ ] The overview shows best test nDCG@10 **0.887** and Recall@5 **0.927**, matching the
      README.
- [ ] Leaderboard: the test split lists 5 rows and the dev split lists 33 rows.
- [ ] Pareto: points are coloured by retrieval mode, and hovering a point shows its config id.
- [ ] Ablations: the rerank factor shows at least one CI that excludes 0.
- [ ] Query explorer: picking a question shows two columns of ranked hits with gold-hit flags.
- [ ] The Methodology page renders both tabs.
- [ ] Add the public URL to `README.md`, then tick this item in `docs/OPEN_QUESTIONS.md`.

## Local equivalents

```bash
make dashboard                                   # http://localhost:8501
docker compose --profile app up -d dashboard     # same, in Docker
docker compose --profile app up -d api           # query API on http://localhost:8000/docs
```
