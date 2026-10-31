# Methodology

> Stub: sections are filled as phases land. Content is written only from
> observed results and verified documentation — never guessed.

## Corpus

- Pinned source, tag/SHA, license, fetch date: see `corpus.lock`.
- Cleaning rules and section parsing: (Phase 1)

## Gold set

- Candidate generation prompt + filters, lexical-overlap threshold used: (Phase 1)
- Human verification protocol, verified count, dev/test split: (Phase 1)

## Chunking

- Token accounting: tiktoken `cl100k_base` (pinned `tiktoken==0.14.0`) measures every size
  bound — `size_tokens`, `overlap_tokens`, `min_tokens`, `max_tokens` — for all three
  chunkers; no whitespace/word-count fallback is used.
- BPE vocabulary cache: `TIKTOKEN_CACHE_DIR` is set (only if unset) to the repo-relative
  `.tiktoken_cache/` before first use, so the vocabulary download lands on the project drive.
- `fixed`: windows of `size_tokens` stepped by `size_tokens - overlap_tokens`, mapped to exact
  char offsets; with zero overlap the windows tile the document contiguously (last window may
  be shorter).
- `recursive_md`: splits only at heading and blank-line paragraph boundaries; budgets are
  measured over the prospective joined slice (`text[start:end]`, separators included), so a
  chunk never exceeds `size_tokens` except when a single indivisible paragraph alone does
  (tolerance: that paragraph is emitted alone as one oversize chunk); `overlap_tokens` applies
  at paragraph boundaries within a section, with the overlap region and the receiving chunk
  both measured on emitted text.
- `semantic`: sentences split with the regex `[^.!?]*[.!?]+|[^.!?]+$` (whitespace-trimmed;
  abbreviations not handled), embedded in one batch via `embed_documents`; breakpoints open
  where consecutive-sentence cosine similarity falls below `threshold`; a single sentence over
  `max_tokens` is force-split into non-overlapping token windows.

## Embedders

- Models and prefix/instruction strings (model cards verified 2026-10-31):
  - `intfloat/e5-small-v2` (https://huggingface.co/intfloat/e5-small-v2): queries
    prefixed `query: `, documents/passages prefixed `passage: ` — both sides required
    per the card's FAQ ("Yes, this is how the model is trained").
  - `BAAI/bge-small-en-v1.5` (https://huggingface.co/BAAI/bge-small-en-v1.5): queries
    prefixed with the instruction `Represent this sentence for searching relevant
    passages: ` (trailing space included, as shown in the card's model list); documents
    unprefixed ("In all cases, no instruction needs to be added to passages").
  - `sentence-transformers/all-MiniLM-L6-v2`
    (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): no prefix on
    either side (plain `model.encode(sentences)` usage).
- All three models are cosine models: vectors are L2-normalized
  (`normalize_embeddings=True`) for documents and queries, dtype `float32`;
  `embed_documents` returns shape `(len(texts), dim)`, `embed_query` returns `(dim,)`.
- Prefixes are applied inside the model-call path; the vector cache is keyed on the
  raw (unprefixed) input text — `(model_name, model_revision, sha256(text))` — so cache
  hits skip prefixing and model compute.
- Vector cache layout: `<cache root>/embeddings/{sha256-of-canonical-key}.npy`, where
  the cache root is `$HF_HOME` when set, else repo-root `.hf_cache` (gitignored); the
  same root is passed to `SentenceTransformer(cache_folder=...)` for model downloads.
- `model_revision` is recorded best-effort as the resolved HF snapshot sha
  (`huggingface_hub.model_info`) at the first embed call; it falls back to the
  requested revision, else `unresolved` (offline).

## Relevance scoring

- Span-overlap rule and minimum-fraction setting: (Phase 3)
- nDCG first-hit-per-gold-span handling: (Phase 3)

## Retrieval index parameters

- HNSW parameters, equalized across pgvector and Qdrant: `m=16`, `ef_construction=64`,
  query-time `hnsw.ef_search=100` (pgvector, Task 2.5; Qdrant parity Task 2.6)
- control_bm25 rationale: separates backend effect from lexical-scorer effect (Phase 2)
- Exact-search (brute-force) sanity check vs ANN: (Phase 3)

## Statistics

- Bootstrap settings (resamples, seed) and paired bootstrap for deltas: (Phase 3)

## Latency and resources

- Warm-up/measure protocol, stage breakdown, hardware: (Phase 3), `results/hardware.json`

## Generation judging

- Judge model, prompt versions, RAGAS version and metric names: (Phase 4)

## Regression gate

- Baseline provenance, threshold, degraded-config failure evidence: (Phase 5)

## Resolved versions

- Python and package pins: `pyproject.toml`, `uv.lock` (recorded here at release)
- pgvector extension: `0.8.7` in image `pgvector/pgvector:0.8.7-pg16-trixie`
  (verified via `SELECT extversion FROM pg_extension WHERE extname='vector'`, Task 2.5)
- Model revisions: (Phase 2)
