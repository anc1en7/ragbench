# SDD ledger — plan: docs/superpowers/plans/2026-10-25-ragbench-implementation.md

Phase: PHASE 2 — Pipeline components (Tasks 2.1–2.10 → Checkpoint 2).
Branch: main (direct, human-consented 2026-10-31). BASE at start: 5f252b0.

## Pre-flight conflict scan (2026-10-31)

| Pair / Task | Produces vs consumes | Finding | Ruling |
|---|---|---|---|
| 2.1 vs Phase 1 T1.2 | Plan T2.1 says "copy spec §4 verbatim"; types.py already exists from Phase 1 | Task is tests-only | Implement tests/unit/test_types.py only; no new types.py (deviation already recorded in plan Phase 1 status) |
| 2.1 chunk_id hash | Types.py Chunk.chunk_id is a plain field; plan T2.1 wants sha256(doc_id,start,end,chunker_config) stability test; no production hash helper exists yet | Helper unspecified | Ruling: T2.1 test uses a local helper encoding the documented rule; production `make_chunk_id()` lands in chunking/__init__.py at T2.3 and its test asserts the same rule |
| 2.3 tokenizer | Plan allows "pinned tokenizer, fallback tiktoken cl100k or whitespace" | Choice needed | Ruling (human): tiktoken cl100k_base, pinned ==, documented in METHODOLOGY |
| 2.3 semantic vs 2.4 | Semantic chunker needs an embedder; T2.4 not built yet | Dependency | Ruling: chunker takes an injected Embedder (types.py protocol); unit tests use fake embedder, no network, no T2.4 dependency |
| 2.4 cache test vs network | "two calls hit disk once" would download a real model | Unit-test purity | Ruling: unit tests use fake embedder for cache logic; real model downloads only in integration tests; HF cache confined to repo .hf_cache/ (human constraint: all large files on D:) |
| 2.5/2.6 vs Makefile test | make test runs tests/unit only; integration needs compose | Verification path | Ruling: integration tests via `uv run pytest tests/integration` with compose up; checkpoint runs both |
| 2.10 fixture "hand-checked" | Plan says write 40 questions with hand-checked gold sections | Human effort | Ruling (human-approved): derive fixture from 157 verified questions + gold spans in a documents.jsonl slice |
| fastembed pin | pyproject has fastembed>=0.9.1; spec rule 3 requires == pins | Violation | Ruling: pin fastembed== in pre-flight alongside new deps |
| Global: downloads on D: | uv/HF/tiktoken default caches live on C: | Human constraint | Ruling: UV_CACHE_DIR=.uv_cache (Makefile export), HF model cache under .hf_cache/, TIKTOKEN_CACHE_DIR in repo; .gitignore += .uv_cache/ |
| 2.9 pipeline vs 2.4–2.8 | Pipeline consumes all protocols | Sequential order | Clean — tasks executed in plan order |

Self-consistency per task: 2.2 grid.yaml skeleton must not invent stages (stages land in 4.1); 2.8 cross-encoder real download integration-only per plan text; 2.9 generate() stays a stub until Phase 4. No other conflicts found.

## Progress

