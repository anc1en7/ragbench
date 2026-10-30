# RAGBench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build RAGBench, a config-driven platform that benchmarks RAG pipelines (chunkers, embedders, retrieval modes, rerankers, two vector backends) on a pinned FastAPI-docs corpus with a human-verified gold set, reporting retrieval quality with bootstrap CIs, latency, and RAGAS generation metrics.

**Architecture:** Offline pipeline (corpus → gold set → indexer → eval harness → staged sweep → `results/`) plus an online FastAPI service sharing the same pipeline code. Fusion (RRF) and reranking run in application code so pgvector and Qdrant are compared with identical logic. A Streamlit dashboard and README generator read only from `results/`.

**Tech Stack:** Python 3.13, uv, typer, pydantic, sentence-transformers, qdrant-client, pgvector/psycopg, fastembed, bm25s, ragas, FastAPI, OpenTelemetry + Arize Phoenix, Streamlit, GitHub Actions, Docker Compose. Host OS: Windows 11 (PowerShell), GNU make via scoop, Docker Desktop.

**Spec:** `RAGBENCH_AGENT_SPEC.md` (repo root). The spec travels with this plan; executors read both. Exact interface shapes are in spec §4; config schema in §5.1; phases/checkpoints in §6.

## Global Constraints

- **Phased execution:** stop at every checkpoint (spec §6), print a short summary (built / how to verify / deviations), and wait for the human to say "continue". Never start the next phase unaided.
- **Never invent results.** Every number in README/dashboard/resume comes from code reading `results/`. Unrun values stay `[X]`.
- **Verify library APIs against current docs before coding against them** (ragas, qdrant-client, sentence-transformers, arize-phoenix, fastembed, pgvector). Pin with `==` in `pyproject.toml`; record resolved versions in `docs/METHODOLOGY.md`.
- **Reproducibility:** seed 42 default, pinned corpus tag in `corpus.lock`, pinned model revisions, hardware in `results/hardware.json`.
- **Small conventional commits:** one logical change, `feat:`/`fix:`/`docs:`/`test:`/`ci:` prefixes.
- **`make lint test` must be green before stopping at any checkpoint.**
- **Secrets:** `.env` is gitignored; `.env.example` holds placeholders only. Never commit keys, model files, or embedding caches.
- **LLM access:** all generation/judging goes through OpenAI-compatible env vars `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (OmniRoute on the human's private VPS). Deviation from spec's Ollama default → log in `docs/OPEN_QUESTIONS.md`.
- **Environment facts:** RTX 3050 Laptop 4 GB VRAM, i5-12450H, 16 GB RAM, Windows, no local LLM. GPU is optional speed-up for embedders/reranker only.
- **Human action items (never do these yourself):** provide OmniRoute URL/key, approve Docker Desktop install, gold-set review, GitHub repo creation approval, dashboard deployment, demo recording, trace screenshot.

---

## PHASE 0 — Scaffold and environment

### Task 0.1: Tool bootstrap (Docker Desktop, GNU make)

**Files:** none in repo (system install)

- [ ] **Step 1: Install GNU make**
  ```powershell
  scoop install make
  make --version   # expect GNU Make 4.x
  ```
- [ ] **Step 2: Install Docker Desktop (may require elevation + reboot — hand to human if winget prompts)**
  ```powershell
  winget install --id Docker.DockerDesktop -e
  ```
  If WSL2 backend missing: `wsl --install` then reboot, then start Docker Desktop.
- [ ] **Step 3: Verify engine**
  ```powershell
  docker info          # no errors
  docker compose version
  ```
- [ ] **Step 4: Commit** (nothing yet — first repo commit comes in Task 0.2)

### Task 0.2: git init, license, ignore rules, env template

**Files:** Create: `.gitignore`, `.env.example`, `LICENSE`

- [ ] **Step 1: Init repo**
  ```powershell
  git init -b main
  ```
- [ ] **Step 2: Write `.gitignore`** — must cover: `.env`, `data/raw/`, `data/*.parquet`, `__pycache__/`, `.venv/`, `*.pyc`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.hf_cache/`, `models/`, `*.bin`, `*.safetensors`, `.streamlit/secrets.toml`, `results/*.parquet` (keep `results/*.json`, `results/summary.csv` committed per spec §3).
- [ ] **Step 3: Write `.env.example`**
  ```env
  # OpenAI-compatible LLM endpoint (OmniRoute). Copy to .env and fill in.
  LLM_BASE_URL=https://your-vps.example.com/v1
  LLM_API_KEY=changeme
  LLM_MODEL=changeme
  # Tracing off by default (spec 5.10)
  TRACING=0
  # Postgres/Qdrant connection (docker-compose defaults)
  PG_DSN=postgresql://ragbench:ragbench@localhost:5432/ragbench
  QDRANT_URL=http://localhost:6333
  ```
- [ ] **Step 4: Write `LICENSE`** — MIT, copyright holder = git `user.name` (read via `git config user.name`; if unset, ask the human).
- [ ] **Step 5: Commit** — `chore: repo init, MIT license, ignore rules, env template`

### Task 0.3: uv project scaffold

**Files:** Create: `pyproject.toml`, `uv.lock`, `src/ragbench/__init__.py`, `src/ragbench/cli.py`, `src/ragbench/hardware.py`, `tests/unit/test_hardware.py`, `tests/__init__.py`

- [ ] **Step 1: Scaffold package**
  ```powershell
  uv init --package --name ragbench --python 3.13
  ```
  Adjust to the spec §3 layout: sources under `src/ragbench/`.
- [ ] **Step 2: Add runtime deps (Phase 0 set), pinning `==` after resolution**
  ```powershell
  uv add typer pydantic pyyaml numpy httpx python-dotenv rich psutil
  uv add --group dev ruff mypy pytest pytest-cov pre-commit
  ```
  Hand-pin the resolved versions in `pyproject.toml` with `==` (spec rule 3).
- [ ] **Step 3: Write `src/ragbench/cli.py`** — typer app with two subcommands now: `hardware` and `llm-probe` (probe implemented in Task 0.7). Entry point in `pyproject.toml`:
  ```toml
  [project.scripts]
  ragbench = "ragbench.cli:app"
  ```
- [ ] **Step 4: Write `src/ragbench/hardware.py`** — detect: platform + python version, `os.cpu_count()`, RAM GB (`psutil`), GPU via `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` subprocess (fall back to `"none"`), CUDA availability if torch importable (it won't be yet — guard the import).
  Produces `results/hardware.json`:
  ```json
  {
    "collected_at": "ISO-8601",
    "platform": "Windows-11-...",
    "python": "3.13.9",
    "cpu": "12th Gen Intel(R) Core(TM) i5-12450H",
    "cpu_cores": 16,
    "ram_gb": 15.7,
    "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "gpu_vram_mb": 4096,
    "llm": {"base_url_host": "", "model": "", "resolved_at": ""},
    "package_versions": {}
  }
  ```
- [ ] **Step 5: Write failing test** `tests/unit/test_hardware.py`:
  ```python
  from ragbench.hardware import collect_hardware

  def test_collect_hardware_shape(tmp_path):
      out = tmp_path / "hardware.json"
      collect_hardware(out)
      import json
      data = json.loads(out.read_text(encoding="utf-8"))
      for key in ("platform", "python", "cpu_cores", "ram_gb", "gpu", "package_versions"):
          assert key in data
  ```
- [ ] **Step 6: Run test, watch it fail** — `uv run pytest tests/unit/test_hardware.py -v` (module missing).
- [ ] **Step 7: Implement minimal `collect_hardware`** until the test passes.
- [ ] **Step 8: Run test green, then `make`-less full check** — `uv run ruff check src tests && uv run pytest -q`.
- [ ] **Step 9: Commit** — `feat: uv project scaffold with hardware detection CLI`

### Task 0.4: lint / type / test configuration

**Files:** Create: `pyproject.toml` [tool.ruff]/[tool.mypy]/[tool.pytest.ini_options] sections, `.pre-commit-config.yaml`

- [ ] **Step 1: Configure ruff** — line-length 100, select `E,F,I,UP,B,SIM`, exclude `data/`.
- [ ] **Step 2: Configure mypy** — `strict = true` for `src/ragbench` only; `ignore_missing_imports = true` for the ML libs not yet installed.
- [ ] **Step 3: Configure pytest** — `testpaths = ["tests"]`, marker `integration` (later used for docker-dependent tests).
- [ ] **Step 4: Write `.pre-commit-config.yaml`** — hooks: `ruff`, `ruff-format`, `mypy`.
- [ ] **Step 5: Run everything** — `uv run ruff check src tests && uv run mypy src && uv run pytest -q` → all green.
- [ ] **Step 6: Commit** — `chore: lint, type-check and test configuration`

### Task 0.5: Makefile

**Files:** Create: `Makefile`

- [ ] **Step 1: Write Makefile** with all spec targets (recipes use `uv run`):
  ```make
  .PHONY: setup lint test up down ingest goldset goldset-review sweep report reproduce dashboard update-baseline

  setup:
  	uv sync --all-extras
  	uv run pre-commit install

  lint:
  	uv run ruff check src tests
  	uv run ruff format --check src tests
  	uv run mypy src

  test:
  	uv run pytest tests/unit -q

  up:
  	docker compose up -d --wait

  down:
  	docker compose down

  ingest:
  	uv run ragbench corpus ingest

  goldset:
  	uv run ragbench goldset generate

  goldset-review:
  	uv run ragbench goldset review

  sweep:
  	uv run ragbench sweep --grid configs/grid.yaml

  report:
  	uv run ragbench report

  reproduce:
  	uv run ragbench reproduce

  dashboard:
  	uv run streamlit run dashboard/app.py

  update-baseline:
  	uv run ragbench eval --update-baseline
  ```
  Targets whose subcommands do not exist yet will fail until their phase lands; Checkpoint 0 only requires `setup lint test up down`.
- [ ] **Step 2: Verify** — `make lint test` green.
- [ ] **Step 3: Commit** — `chore: Makefile with full target surface`

### Task 0.6: docker-compose (pgvector, Qdrant, optional Phoenix)

**Files:** Create: `docker-compose.yml`

- [ ] **Step 1: Write compose file**
  ```yaml
  name: ragbench
  services:
    pgvector:
      image: pgvector/pgvector:pg16
      environment:
        POSTGRES_USER: ragbench
        POSTGRES_PASSWORD: ragbench
        POSTGRES_DB: ragbench
      ports: ["5432:5432"]
      volumes: ["pgdata:/var/lib/postgresql/data"]
      healthcheck:
        test: ["CMD-SHELL", "pg_isready -U ragbench"]
        interval: 5s
        timeout: 3s
        retries: 10
    qdrant:
      image: qdrant/qdrant:v1.12.4   # verify current stable tag at build time; pin
      ports: ["6333:6333", "6334:6334"]
      volumes: ["qdrant_data:/qdrant/storage"]
      healthcheck:
        test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
        interval: 5s
        timeout: 3s
        retries: 10
    phoenix:
      image: arizephoenix/phoenix:latest
      profiles: ["phoenix"]
      ports: ["6006:6006"]
  volumes:
    pgdata: {}
    qdrant_data: {}
  ```
  (Verify current image names/tags against Docker Hub before committing — spec rule 3.)
- [ ] **Step 2: Verify** — `docker compose up -d --wait` → both core services healthy; `docker compose ps` shows healthy.
- [ ] **Step 3: Commit** — `chore: docker compose with pgvector and qdrant`

### Task 0.7: OmniRoute endpoint probe

**Files:** Create: `src/ragbench/llm_client.py`, `tests/unit/test_llm_client.py`; Modify: `src/ragbench/cli.py`

- [ ] **Step 1: Ask the human for `LLM_BASE_URL` and `LLM_API_KEY`** (checkpoint interaction — write them into `.env` locally, never commit).
- [ ] **Step 2: Write failing test** — `test_llm_client.py`: `probe_endpoint(base_url, key)` returns `{"models": [...], "chat_ok": bool}` using `httpx` with a mocked transport (no network in unit tests).
- [ ] **Step 3: Implement `llm_client.py`** — `OpenAICompatClient` with `chat(messages, temperature=0.0) -> str`, `models() -> list[str]`; reads env vars, raises a clear error naming the missing var. Uses `httpx.Client(timeout=120)`.
- [ ] **Step 4: Wire `ragbench llm-probe`** — probes `GET {LLM_BASE_URL}/models` and sends a 5-token chat completion; resolves the model id, then **merges into `results/hardware.json`** under `llm` (base_url_host only, never the key).
- [ ] **Step 5: Run live** — `uv run ragbench llm-probe` against the VPS; confirm `results/hardware.json` records the fixed model id. **That model is now pinned for generation + judging** (spec §5.5).
- [ ] **Step 6: Commit** — `feat: OpenAI-compatible LLM client with endpoint probe`

### Task 0.8: CI skeleton + OPEN_QUESTIONS

**Files:** Create: `.github/workflows/ci.yml`, `docs/OPEN_QUESTIONS.md`, `docs/METHODOLOGY.md` (stub sections), `docs/LIMITATIONS.md` (stub sections)

- [ ] **Step 1: Write `ci.yml`** — job `lint-test` on `ubuntu-latest`: checkout, `astral-sh/setup-uv` with Python 3.13, cache `~/.cache/uv`, `uv sync`, `make lint test`. (Regression + docker jobs arrive in Phase 5.)
- [ ] **Step 2: Seed `docs/OPEN_QUESTIONS.md`** with: LLM endpoint = OmniRoute VPS (deviation from Ollama, §2), Docker Desktop used instead of plain docker engine on Windows, pending: GitHub repo creation (human).
- [ ] **Step 3: Add METHODOLOGY/LIMITATIONS stubs** with section headings only (no invented content).
- [ ] **Step 4: Commit** — `ci: lint-test workflow; docs: open questions and doc stubs`

### Task 0.9: Checkpoint 0 verification

- [ ] **Step 1:** `make setup lint test` → green.
- [ ] **Step 2:** `docker compose up -d --wait` → pgvector + qdrant healthy.
- [ ] **Step 3:** `results/hardware.json` exists with GPU, CPU, RAM, pinned LLM model.
- [ ] **Step 4: STOP.** Print Checkpoint 0 summary: what was built, how to verify, detected hardware, chosen model sizes, deviations. Wait for "continue".

---

## PHASE 1 — Corpus and gold set

> **Execution status (2026-10-29):** **Phase 1 COMPLETE — Checkpoint 1 cleared.** Tasks 1.1–1.6 done (TDD, committed, `make lint test` green — 65 tests). Final artifacts: `corpus.lock` (FastAPI 0.162.0, MIT), `data/documents.jsonl` (156 docs, 2,199 sections), `data/goldset/candidates.jsonl` (483 candidates from 506 sampled; drops: 9 generation errors, 5 judge-no, 9 duplicates, 0 heading/lexical), **157 human-verified** (1 rejected, 18 flagged), `data/goldset/splits.json` (seed 42, dev 47 / test 110, stratified by type; committed via .gitignore whitelist). LLM: `LLM_MODEL=rag` (user's OmniRoute combo: Mistral Ministral 14B / Grok pool), verified with a 15/15 burst probe and a full generation run at 0.5s request pacing (`--pacer`); Grok Build `gc/*` works post-OmniRoute-update but trips xAI `model_cooldown` 429s under burst (handled via `RateLimitError.retry_after` + thread-safe pacer in `goldset/generate.py`). Deviations: `src/ragbench/types.py` created in 1.2 (spec §4 verbatim — Task 2.1 becomes tests-only); shared `goldset/schema.py` for generate/review/split; review tool writes resumable `review_log.jsonl`; blank-run collapse "2+ blank lines → 1"; `MIN_SECTION_TOKENS=80` (corpus median section is 51 tokens — 120 left only 277 eligible); dedup embedder = fastembed ONNX `bge-small-en-v1.5` (no torch). Awaiting human "continue" to start Phase 2.

### Task 1.1: Fetch pinned corpus + corpus.lock

**Files:** Create: `src/ragbench/corpus/fetch.py`, `corpus.lock`, `tests/unit/test_fetch.py`

- [ ] **Step 1: Resolve pin** — query GitHub API for the latest stable `fastapi/fastapi` release tag (or newest tag if no releases); record tag + commit SHA + fetch date.
- [ ] **Step 2: Implement fetch** — download release tarball via `httpx`, extract only `docs/en/docs/**/*.md` into `data/raw/` (gitignored). Verify `LICENSE` of fastapi repo (MIT) permits reuse; record in `corpus.lock`.
- [ ] **Step 3: Write `corpus.lock`** (committed):
  ```json
  {"repo": "https://github.com/fastapi/fastapi", "tag": "<resolved>", "commit_sha": "<resolved>",
   "docs_path": "docs/en/docs", "license": "MIT", "fetched_at": "<ISO>", "file_count": <n>,
   "sha256_of_files": "<hash>"}
  ```
- [ ] **Step 4: Test** — fetch is idempotent (second run no-ops when lock matches); test uses a tiny mocked tarball fixture.
- [ ] **Step 5: Commit** — `feat: pinned fastapi docs fetcher with corpus.lock`

### Task 1.2: Clean markdown + section parsing → documents.jsonl

**Files:** Create: `src/ragbench/corpus/clean.py`, `src/ragbench/corpus/sections.py`, `src/ragbench/corpus/build.py`, `tests/unit/test_sections.py`

- [ ] **Step 1: Write failing tests for section parsing** — a markdown string with `#`, `##`, `###` headings must yield `Section` objects (spec §4 shape) whose `[start, end)` slices satisfy `doc.text[start:end]` covering exactly that heading's body, `section_id == f"{doc_id}#{slug}"`, and heading_path reflects ancestor headings. Also: offsets survive cleaning (compute offsets against the *cleaned* text).
- [ ] **Step 2: Implement `clean.py`** — strip YAML front matter, normalize line endings, collapse >2 blank lines, remove nav-only includes if present; keep code fences intact.
- [ ] **Step 3: Implement `sections.py`** — heading regex over lines (`^(#{1,6})\s+(.*)$`), state machine for heading stack, slugify (lowercase, non-alnum → `-`), compute char offsets on cleaned text.
- [ ] **Step 4: Implement `build.py`** — walk `data/raw/**/*.md`, write `data/documents.jsonl` with `{"doc_id", "title", "text", "sections":[...]}`.
- [ ] **Step 5: Run tests green** → `uv run pytest tests/unit/test_sections.py -v`.
- [ ] **Step 6: Commit** — `feat: corpus cleaning and section parser`

### Task 1.3: Gold-set generation with quality filters

**Files:** Create: `src/ragbench/goldset/generate.py`, `src/ragbench/goldset/filters.py`, `src/ragbench/goldset/prompts.py`, `tests/unit/test_filters.py`

- [ ] **Step 1: Write failing tests for filters** (pure functions, no network):
  - `heading_copy_ratio(question, heading)` — drops when question is near-copy of heading (threshold 0.6 token overlap; **record the threshold** in output metadata).
  - `lexical_overlap(question, section_text)` — drop above threshold (protects BM25 from unfair questions).
  - `dedup(embeddings, pairs, threshold=0.92)` — cosine-similarity dedup.
- [ ] **Step 2: Implement sampling** — stratified over documents and length buckets; skip sections < 120 tokens or whose path matches changelog/tutorial-index patterns; skip sections that are >60% code fences.
- [ ] **Step 3: Implement prompts + LLM calls** — one question + reference answer per section, typed `factual | how_to | why_when`; `temperature=0.0` through `OpenAICompatClient`. Concurrency capped (e.g. 4) with retry/backoff.
- [ ] **Step 4: Implement answerability judge** — second prompt: "Is the reference answer fully supported by the section? yes/no" → drop `no` (and record drop counts per filter stage).
- [ ] **Step 5: Run generation** — `uv run ragbench goldset generate` until ≥ 400 survivors land in `data/goldset/candidates.jsonl` with fields per spec §5.6: `qid, question, reference_answer, gold_sections[], gold_spans[], question_type, source_doc, verified: false`.
- [ ] **Step 6: Commit** — `feat: gold set generation with quality filters`

### Task 1.4: Human review tool (terminal)

**Files:** Create: `src/ragbench/goldset/review.py`, `tests/unit/test_review.py`

- [ ] **Step 1: Implement** `ragbench goldset review` — rich-based prompt loop per candidate: show question, reference answer, section text excerpt with gold span highlighted; keys `a`ccept / `e`dit (opens line editor) / `r`eject / `f`lag / `q`uit. Resumable (tracks last qid). Writes `data/goldset/verified.jsonl` with `verified: true`.
- [ ] **Step 2: Test** — feed a scripted stdin fixture; assert accepted/rejected items land in the right file.
- [ ] **Step 3: Commit** — `feat: terminal gold-set review tool`

### Task 1.5: Splits

**Files:** Create: `src/ragbench/goldset/split.py`, `tests/unit/test_split.py`

- [ ] **Step 1: Write failing test** — given 10 fake verified items across 3 types, split produces dev ≈ 30% / test ≈ 70%, stratified per type, deterministic given `seed=42`, ids match `splits.json` shape `{"seed": 42, "dev": [...], "test": [...]}`.
- [ ] **Step 2: Implement + run on real verified set** → `data/goldset/splits.json` (committed).
- [ ] **Step 3: Commit** — `feat: seeded stratified dev/test split`

### Task 1.6: Checkpoint 1 (USER ACTION)

- [ ] **Step 1:** `make lint test` green; `make goldset` produced candidates.
- [ ] **Step 2: STOP and print:** corpus stats (documents, sections, tokens), filter drop counts per stage, 10 sample candidates, verified count so far.
- [ ] **Step 3: Wait for human.** Either they verify ≥ 150 questions via `make goldset-review`, or they explicitly authorize provisional mode — in which case mark every downstream output `"provisional: unverified gold set"` and report verified-only vs all-synthetic metrics separately (spec §5.6).

---

## PHASE 2 — Pipeline components

### Task 2.1: Core types (exact spec §4 shape)

**Files:** Create: `src/ragbench/types.py`, `tests/unit/test_types.py`

- [ ] **Step 1: Copy the dataclasses/Protocols from spec §4 verbatim** (`Document`, `Section`, `Chunk`, `Hit`, `Chunker`, `Embedder`, `Store`, `Reranker`).
- [ ] **Step 2: Tests** — frozen dataclasses reject mutation; `chunk_id` is a stable sha256 of `(doc_id, start, end, chunker_config)` (assert equality across two constructions, inequality when config differs).
- [ ] **Step 3: Commit** — `feat: core type contracts`

### Task 2.2: Config models

**Files:** Create: `src/ragbench/config.py`, `configs/pipeline.example.yaml`, `configs/grid.yaml` (skeleton), `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests** — load `configs/pipeline.example.yaml` (content from spec §5.1) into pydantic models; reject unknown `retrieval.mode`; `config_id` equals short hash of normalized (sorted, default-filled) config and is stable across dict ordering.
- [ ] **Step 2: Implement models** — `ChunkerCfg, EmbedderCfg, RetrievalCfg, RerankCfg, GenerationCfg, PipelineConfig` with `config_id` property.
- [ ] **Step 3: Commit** — `feat: pydantic pipeline configuration`

### Task 2.3: Chunkers (fixed, recursive_md, semantic)

**Files:** Create: `src/ragbench/chunking/fixed.py`, `recursive_md.py`, `semantic.py`, `src/ragbench/chunking/__init__.py` (registry), `tests/unit/test_chunking.py`

- [ ] **Step 1: Write failing invariant tests** (shared parametrized suite over all 3 chunkers):
  - `0 <= start < end <= len(doc.text)` and `chunk.text == doc.text[start:end]`.
  - fixed: token windows of size/overlap per config (tokens approximated by a pinned tokenizer — verify availability; fallback to `tiktoken` cl100k or whitespace-word counting, **documented** in METHODOLOGY).
  - recursive_md: splits only at heading/paragraph boundaries, ≤ size + tolerance, overlap ≤ configured.
  - semantic: min/max bounds respected; breakpoints from consecutive-sentence cosine similarity using a small embedder (inject a fake embedder in unit tests — no network).
- [ ] **Step 2: Implement each chunker** against `Chunker` protocol; chunk metadata includes heading path for `recursive_md`.
- [ ] **Step 3: Registry** — `get_chunker(cfg: ChunkerCfg) -> Chunker`.
- [ ] **Step 4: Commit** — `feat: fixed, recursive markdown and semantic chunkers`

### Task 2.4: Embedders with disk cache

**Files:** Create: `src/ragbench/embedding/base.py`, `st_embedder.py`, `cache.py`, `tests/unit/test_cache.py`, `tests/unit/test_prefixes.py`

- [ ] **Step 1: Write failing tests** — cache key = `(model_name, model_revision, sha256(text))`; two calls with same text hit disk once (assert via counter); prefix rules: e5 → `"query: "` / `"passage: "`, bge → query instruction prefix, MiniLM → none.
- [ ] **Step 2: Verify each model card** (bge-small-en-v1.5, all-MiniLM-L6-v2, e5-small-v2) for exact prefix/instruction strings — record URLs in METHODOLOGY.
- [ ] **Step 3: Implement** — `SentenceTransformerEmbedder(name, revision)` implementing `Embedder` protocol; `embed_documents` batches with `normalize_embeddings=True` for cosine models; cache in `.hf_cache/embeddings/{sha}.npy` (gitignored); record resolved `model_revision` (snapshot commit from HF).
- [ ] **Step 4: Commit** — `feat: sentence-transformers embedders with keyed disk cache`

### Task 2.5: pgvector store

**Files:** Create: `src/ragbench/stores/pgvector_store.py`, `tests/integration/test_pgvector.py`

- [ ] **Step 1: Verify current pgvector + psycopg APIs** against docs (spec rule 3).
- [ ] **Step 2: Write failing integration test** (marked `integration`, requires `make up`): reset namespace → add 100 tiny chunks + vectors → `dense_search` returns expected neighbor order for a planted vector; `sparse_search` (tsvector + `ts_rank_cd`, `websearch_to_tsquery`) finds a keyword query's doc; `size_bytes` > 0. DDL: `CREATE EXTENSION vector;` table `chunks(namespace, chunk_id, doc_id, start, end, text, embedding vector(dim))`, HNSW cosine `m=16, ef_construction=64`; session `SET hnsw.ef_search=100`. GIN index on `to_tsvector('english', text)`.
- [ ] **Step 3: Implement `PgvectorStore(Store)`** — connection from `PG_DSN`; creates schema lazily on `reset`.
- [ ] **Step 4: Run** — `uv run pytest tests/integration/test_pgvector.py -v` green with compose up.
- [ ] **Step 5: Commit** — `feat: pgvector store with HNSW dense and FTS sparse search`

### Task 2.6: Qdrant store

**Files:** Create: `src/ragbench/stores/qdrant_store.py`, `tests/integration/test_qdrant.py`

- [ ] **Step 1: Verify current `qdrant-client` + `fastembed` sparse-vector APIs** (these change often).
- [ ] **Step 2: Write failing integration test** — mirror the pgvector test: dense HNSW cosine `m=16, ef_construct=64, hnsw_ef=100`; sparse BM25-style search via fastembed's sparse encoder; `size_bytes` from collection info (payload + vectors on disk, documented approximation).
- [ ] **Step 3: Implement `QdrantStore(Store)`** with named namespaces per config.
- [ ] **Step 4: Run integration test green.**
- [ ] **Step 5: Commit** — `feat: qdrant store with HNSW dense and sparse BM25 search`

### Task 2.7: control_bm25 (backend-agnostic)

**Files:** Create: `src/ragbench/stores/control_bm25.py`, `tests/unit/test_control_bm25.py`

- [ ] **Step 1: Test** — index 5 toy docs; query ranks the doc containing the term first; identical results across runs (pure in-process, `bm25s` preferred — verify API, else `rank_bm25`).
- [ ] **Step 2: Implement** as a `Store`-shaped object where `dense_search` raises `NotImplementedError` (only `sparse_search` is meaningful) — documented.
- [ ] **Step 3: Commit** — `feat: in-process BM25 control backend`

### Task 2.8: RRF fusion + rerankers

**Files:** Create: `src/ragbench/retrieval/rrf.py`, `src/ragbench/rerank/none.py`, `src/ragbench/rerank/cross_encoder.py`, `tests/unit/test_rrf.py`, `tests/unit/test_rerank.py`

- [ ] **Step 1: Write failing RRF test with hand-computed expected order** — `score(c) = Σ 1/(60 + rank_i)` over the dense and sparse lists; ties broken by stable chunk_id sort; output `Hit.source == "fused"`.
- [ ] **Step 2: Implement `fuse(lists: list[list[Hit]], k=60)`** — in application code (spec §4 design rule), identical for both backends.
- [ ] **Step 3: Write failing reranker tests** — `none` passes through with normalized scores; `cross_encoder` with a stub model returns sorted-by-score descending, top_n → final_k truncation, `source == "reranked"`.
- [ ] **Step 4: Implement** `CrossEncoderReranker` using `sentence_transformers.CrossEncoder("BAAI/bge-reranker-base")` (real model download runs in integration test only; unit tests stub it).
- [ ] **Step 5: Commit** — `feat: RRF fusion and cross-encoder reranker`

### Task 2.9: Pipeline assembly + `ragbench query` CLI

**Files:** Create: `src/ragbench/pipeline.py`, extend `src/ragbench/cli.py`, `tests/integration/test_query.py`

- [ ] **Step 1: Write failing integration test** — build mini index on both backends, run pipeline on one question, assert: ranked hits with scores, `timings_ms` keys `embed_query, dense, sparse, fusion, rerank, total`, sources labeled correctly.
- [ ] **Step 2: Implement `Pipeline`** — `retrieve(question, cfg) -> list[Hit]` with per-stage `time.perf_counter()` timings; `generate()` optional (Phase 4 wires RAGAS; generation prompt stub here per spec §`generation/`).
- [ ] **Step 3: Wire CLI** — `ragbench query --config configs/pipeline.example.yaml "How do I declare a query parameter?"` prints chunks (doc_id, score, source) + timings table.
- [ ] **Step 4: Commit** — `feat: pipeline assembly and single-query CLI`

### Task 2.10: Mini fixture + Checkpoint 2

**Files:** Create: `tests/fixtures/mini_corpus/` (~30 docs subset of corpus + ~40 questions with gold spans)

- [ ] **Step 1:** Slice ~30 small docs + write 40 questions with hand-checked gold sections into fixture files (committed; used by CI regression in Phase 5).
- [ ] **Step 2: Checkpoint 2:** run the spec's sample query on **both** backends, show ranked chunks + timings; `make lint test` green (including integration with compose up). **STOP, print summary, wait for "continue".**

---

## PHASE 3 — Evaluation harness

### Task 3.1: Span-overlap scorer

**Files:** Create: `src/ragbench/eval/spans.py`, `tests/unit/test_spans.py`

- [ ] **Step 1: Write failing tests with hand-worked examples** — hit if overlap ≥ max(20% of chunk length, 20% of gold span) (configurable `--min-overlap-frac`, default 0.2); boundary cases: zero overlap, partial, contained, multiple gold spans; a chunk overlapping 2 spans counts each span at most once (first hit only, per spec §5.7 nDCG rule).
- [ ] **Step 2: Implement** `is_hit(chunk, gold_spans, min_frac) -> bool` and `span_hits(ranked_hits, gold_spans) -> list[bool]`.
- [ ] **Step 3: Commit** — `feat: chunker-agnostic span-overlap relevance scoring`

### Task 3.2: Retrieval metrics + bootstrap CIs

**Files:** Create: `src/ragbench/eval/retrieval_metrics.py`, `src/ragbench/eval/stats.py`, `tests/unit/test_retrieval_metrics.py`, `tests/unit/test_stats.py`

- [ ] **Step 1: Write failing tests with hand-computed examples** — 3-query fixture where Recall@{1,3,5,10}, MRR@10, nDCG@10 are computed by hand in the test; assert exact equality (float tol 1e-9).
- [ ] **Step 2: Implement metrics** per query → aggregate mean.
- [ ] **Step 3: Write failing CI tests** — seeded bootstrap (1000 resamples, `seed=42`) returns `(lo, hi)` with `lo <= mean <= hi`; paired bootstrap on per-question differences returns CI; deterministic across runs.
- [ ] **Step 4: Implement `bootstrap_ci(values, n=1000, seed=42)` and `paired_bootstrap_ci(a, b, ...)`.**
- [ ] **Step 5: Commit** — `feat: retrieval metrics with bootstrap confidence intervals`

### Task 3.3: Latency + resource measurement

**Files:** Create: `src/ragbench/eval/latency.py`, `tests/unit/test_latency.py`

- [ ] **Step 1: Test** — profiler records per-stage durations, discards pass 0 (warm-up), computes p50/p95 from ≥3 passes over given query list.
- [ ] **Step 2: Implement** — also capture index wall-time, chunks/sec throughput, `store.size_bytes`, peak RAM via `psutil` during indexing, tokens/query for generation runs (from prompt+context token counts).
- [ ] **Step 3: Commit** — `feat: latency profiler and indexing resource metrics`

### Task 3.4: Results schema + writer

**Files:** Create: `src/ragbench/eval/results.py`, `tests/unit/test_results.py`

- [ ] **Step 1: Write failing test** — a constructed run result round-trips through `RunResult` pydantic model to `results/runs/{config_id}.json` and validates; includes config, `config_id`, seed, corpus SHA (from `corpus.lock`), package versions, hardware ref, split, metrics + CIs, latencies, index stats.
- [ ] **Step 2: Implement writer + `results/summary.csv` appender** (header written once; idempotent re-run updates the row for the same `config_id` rather than duplicating).
- [ ] **Step 3: Commit** — `feat: schema-validated run results and summary.csv writer`

### Task 3.5: `ragbench eval` + ANN exact-search sanity check

**Files:** Create: extend `src/ragbench/cli.py`, `tests/integration/test_eval_dev.py`

- [ ] **Step 1: Write failing integration test** — evaluate baseline dense config (spec §5.1 example) on the **dev** split with mini fixture; assert metrics + CIs present, results file written and schema-valid.
- [ ] **Step 2: Implement CLI** — `ragbench eval --config ... --split dev|test [--update-baseline]`.
- [ ] **Step 3: Exact-search check** — brute-force cosine over all vectors vs HNSW result on a sample; report recall@10 agreement (should be ≥ 0.98) in the run JSON field `ann_sanity_recall`.
- [ ] **Step 4: Checkpoint 3:** print baseline dev metrics + CIs + latency table; unit tests prove metrics against hand examples. **STOP, summary, wait for "continue".**

---

## PHASE 4 — Sweep and generation metrics

### Task 4.1: Staged grid definition

**Files:** Modify: `configs/grid.yaml`

- [ ] **Step 1: Encode stages A–E from spec §5.8** — A: chunkers × sizes × embedders (dense, pgvector); B: top-3 from A × {dense, sparse, hybrid, control_bm25}; C: top-3 from B × {none, cross-encoder} × top_n {20, 50}; D: top-3 from C × {pgvector, qdrant}; E: top-5 × generation.
- [ ] **Step 2: Commit** — `feat: staged sweep grid definition`

### Task 4.2: Sweep runner (estimate, cache, resume)

**Files:** Create: `src/ragbench/sweep/runner.py`, `src/ragbench/sweep/stages.py`, `tests/unit/test_sweep_runner.py`

- [ ] **Step 1: Write failing tests** — (a) `estimate(grid)` returns exact run count per stage before executing; (b) resume: pre-create a completed run JSON, runner skips it; (c) stage selection picks top-N by dev Recall@5 from summary.
- [ ] **Step 2: Implement runner** — writes one JSON per run + appends `results/summary.csv`; records config, seed, corpus SHA, package versions, hardware in every run; embeddings/indexes cached by `config_id`; `KeyboardInterrupt` leaves a resumable state (`.sweep_state.json`).
- [ ] **Step 3: Commit** — `feat: staged sweep runner with caching and resume`

### Task 4.3: RAGAS runner (top configs)

**Files:** Create: `src/ragbench/eval/ragas_runner.py`, `tests/unit/test_ragas_wrapper.py`

- [ ] **Step 1: Verify current RAGAS API** (metric names + LLM wrapper changed across versions — check current docs, pin `ragas==`).
- [ ] **Step 2: Wrap `OpenAICompatClient` as RAGAS LLM**; temperature 0; fixed seed where supported; record judge model + prompt versions in run JSON.
- [ ] **Step 3: Implement** — on top-5 configs × 100 fixed test questions: faithfulness, answer relevancy, context precision, context recall. Write to run JSONs + summary columns.
- [ ] **Step 4: Prepare `docs/judge_spotcheck.md`** — 20 auto-exported judged items (question, reference, contexts, judge verdicts) left unchecked for the human (spec §5.7).
- [ ] **Step 5: Commit** — `feat: RAGAS generation metrics on finalist configs`

### Task 4.4: dev selection → final test evaluation + leaderboard

**Files:** Create: `src/ragbench/sweep/select.py`, extend CLI

- [ ] **Step 1:** select winners on **dev only**, then run those finalists on **test** (leakage rule, spec §7).
- [ ] **Step 2:** print leaderboard from `results/summary.csv` (test rows): Recall@5, MRR@10, nDCG@10 + CIs, p95 latency.
- [ ] **Step 3:** findings summary draft **generated** by code from `results/` (no hand-written numbers), suspicious results flagged as `hypothesis:` lines.
- [ ] **Step 4: Checkpoint 4 (USER REVIEW):** report run counts, total runtime, leaderboard, top configs with CIs, flagged surprises. **STOP — human reviews before any README work.**

---

## PHASE 5 — API, tracing, CI, dashboard

### Task 5.1: FastAPI service

**Files:** Create: `src/ragbench/api/app.py`, `schemas.py`, `pipeline_cache.py`, `tests/integration/test_api.py`, `Dockerfile.api`

- [ ] **Step 1: Write failing httpx tests** (against `TestClient`, mini fixture): `/health` returns service + both backend connectivity; `/configs` lists configs from `results/` + `configs/`; `/query` with `{question, config_id, generate}` returns contexts + `timings_ms` + `trace_id`; input over length limit → 422 with clear message; unknown `config_id` → 404 with clear message; `/eval/summary` returns summary.csv as JSON.
- [ ] **Step 2: Implement app** — pydantic request/response models, pipelines cached per `config_id` (LRU), OpenAPI at `/docs`.
- [ ] **Step 3: `Dockerfile.api`** — python:3.13-slim + uv; compose service `api` with env from `.env`.
- [ ] **Step 4: Commit** — `feat: FastAPI query service with OpenAPI docs`

### Task 5.2: OpenTelemetry tracing → Phoenix

**Files:** Create: `src/ragbench/tracing/setup.py`; Modify: `src/ragbench/pipeline.py`, `.env.example`

- [ ] **Step 1:** spans around each pipeline stage (embed, dense, sparse, fusion, rerank, generate); **off unless `TRACING=1`**; exporter to Phoenix (compose profile `phoenix`); zero overhead when disabled (test asserts no span exporter configured).
- [ ] **Step 2: Commit** — `feat: OpenTelemetry stage tracing behind TRACING flag`

### Task 5.3: CI regression gate + baseline

**Files:** Create: `eval/baseline.json`, `configs/regression_thresholds.json`, Modify: `.github/workflows/ci.yml`, extend CLI

- [ ] **Step 1:** generate baseline from mini fixture: `make update-baseline` → `eval/baseline.json` (Recall@5, MRR@10 with tiny embedding model, CPU).
- [ ] **Step 2: CI job `regression-eval`** — service containers `pgvector/pgvector:pg16` + `qdrant/qdrant`, HF model cache action, runs eval on mini corpus; **fails if Recall@5 drops > 0.03 absolute** (threshold in `configs/regression_thresholds.json`); target < 10 min.
- [ ] **Step 3: CI job `docker-build`** — builds API + dashboard images.
- [ ] **Step 4: Negative evidence** — run locally with a deliberately degraded config (chunk size 16) and capture the regression failure output into `docs/METHODOLOGY.md` (spec Checkpoint 5).
- [ ] **Step 5: Commit** — `ci: regression gate with baseline and degraded-config evidence`

### Task 5.4: Streamlit dashboard (replay mode)

**Files:** Create: `dashboard/app.py`, `dashboard/pages/1_Leaderboard.py`, `2_Pareto.py`, `3_Ablations.py`, `4_Query_Explorer.py`, `5_Methodology.py`, `Dockerfile.dashboard`

- [ ] **Step 1: Leaderboard** — sortable/filterable by chunker, embedder, mode, reranker, backend; columns Recall@5, MRR@10, nDCG@10 (with CI strings), p95 latency; reads `results/summary.csv` + run JSONs only (replay).
- [ ] **Step 2: Pareto** — Recall@5 vs p95 latency scatter, colored by mode, frontier highlighted, log-x option, selection shows config.
- [ ] **Step 3: Ablations** — per-factor effects with CIs + paired-bootstrap deltas vs baseline config.
- [ ] **Step 4: Query explorer** — pick gold question + two configs; side-by-side retrieved chunks with gold span highlighted, scores, stage timings from stored per-query results (keep per-query test results small enough to commit, else release asset per spec §5.12).
- [ ] **Step 5: Methodology page** — renders `docs/METHODOLOGY.md` + `docs/LIMITATIONS.md`.
- [ ] **Step 6: Commit** — `feat: streamlit replay dashboard`

### Task 5.5: Deployment docs

**Files:** Create: `docs/DEPLOY.md`

- [ ] **Step 1:** step-by-step for Hugging Face Spaces / Streamlit Community Cloud: repo contents, secrets (`results/` present), one-click buttons, verification checklist. **Human performs the login/deploy (spec §5.12).**
- [ ] **Step 2: Commit** — `docs: dashboard deployment guide`

- [ ] **Checkpoint 5:** `docker compose up` serves API; sample `curl` works; CI passes on a PR; degraded config fails regression (evidence saved); dashboard runs locally in replay mode. **STOP — human deploys dashboard. Wait for "continue".**

---

## PHASE 6 — Docs, polish, release

### Task 6.1: README generator

**Files:** Create: `src/ragbench/report/readme.py`, `src/ragbench/report/figures.py`, Modify: `README.md`

- [ ] **Step 1: `README.md` skeleton** per spec §8 with marker comments `<!-- BEGIN:RESULTS TABLE -->` / `<!-- END -->`, `<!-- BEGIN:FINDINGS -->` / `<!-- END -->`, and `[X]` placeholders everywhere a number belongs.
- [ ] **Step 2: `make report`** — code generates: 4–6 findings bullets (each with number + CI from `results/`), results table, Pareto PNG; injects between markers. If a run hasn't happened → emits literal `not yet run`, never a guess.
- [ ] **Step 3: Test** — generator on a fixture `results/` produces deterministic markdown; empty results → `not yet run` strings.
- [ ] **Step 4: Commit** — `feat: auto-generated README results and figures`

### Task 6.2: Methodology, limitations, architecture, storyboard

**Files:** Modify: `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`; Create: `docs/architecture.md`, `docs/DEMO_STORYBOARD.md`

- [ ] **Step 1: METHODOLOGY** — corpus pin, section/overlap rules + thresholds used, HNSW params equalized, control_bm25 rationale, bootstrap method, latency protocol, resolved package/model versions, degraded-config regression evidence, judge model + prompt versions.
- [ ] **Step 2: LIMITATIONS** — judge-model bias (same endpoint for generate/judge — self-preference risk), synthetic questions with human verification, one corpus, one hardware profile, ANN-vs-exact approximation, GPU-tier caveats. No "state of the art" anywhere (spec §7).
- [ ] **Step 3: architecture.md** — mermaid diagram from spec §3, kept current.
- [ ] **Step 4: DEMO_STORYBOARD.md** — shot list for the human's screen recording.
- [ ] **Step 5: Commit** — `docs: methodology, limitations, architecture, demo storyboard`

### Task 6.3: `make reproduce`

**Files:** Create: `src/ragbench/reproduce.py` + CLI wiring

- [ ] **Step 1:** one command that from clean clone: setup → fetch corpus (lock-verified) → build documents → (gold set from committed verified.jsonl) → index → eval → sweep fast-path → report regeneration. Full sweep behind `--full`; default documented fast path (mini fixture + top configs) with expected runtime/hardware stated in README.
- [ ] **Step 2: Commit** — `feat: reproduce entrypoint`

### Task 6.4: Release + resume bullets

- [ ] **Step 1:** commit larger result files as a GitHub release asset (`gh release create`); README links it.
- [ ] **Step 2: Write `docs/resume_bullets.md`** — spec §9 templates filled **only** with numbers read from `results/summary.csv`; any missing → `[X]`.
- [ ] **Step 3: Commit** — `docs: release notes and resume bullets from real results`

### Task 6.5: FINAL_REPORT + Checkpoint 6 (final)

**Files:** Create: `docs/FINAL_REPORT.md`

- [ ] **Step 1:** what was built; exact commands; resolved package + model versions; hardware; total compute time; known issues; deviations from spec with reasons; remaining human actions (gold-set review if incomplete, dashboard deploy, demo recording, trace screenshot).
- [ ] **Step 2: Clean-clone test** — clone repo to temp dir, run `make setup reproduce` (fast path), confirm outputs match committed results within stated noise.
- [ ] **Step 3: STOP.** Final summary; human records demo + captures one Phoenix trace screenshot into `docs/`.

---

## Self-Review (spec coverage)

- Spec §1 success criteria → Phases 0–6 tasks above; `make reproduce` in 6.3; 150-question gold set gated at Checkpoint 1; staged sweep at 4.1–4.4; metrics at 3.2; API at 5.1; CI gate at 5.3; dashboard at 5.4; auto-generated README at 6.1; MIT + storyboard at 0.2/6.2.
- Spec §4 interfaces → Task 2.1 verbatim; fusion-in-app-code rule → 2.8; cache key rule → 2.4.
- Spec §5.x → each covered in the phase-owning task (config 2.2, chunkers 2.3, stores 2.5–2.7, rerank 2.8, models/hardware 0.3/0.7, gold set 1.1–1.5, metrics 3.1–3.3, sweep 4.1–4.4, API 5.1, tracing 5.2, CI 5.3, dashboard 5.4).
- Spec §0 working rules → Global Constraints.
- Placeholders: none introduced; unresolved facts (exact FastAPI tag, pinned package versions, Qdrant image tag, model revisions) are **resolved at build time by named steps**, which is required by spec rules 2–3.
