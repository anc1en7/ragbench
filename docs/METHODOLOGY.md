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

- Relevance is defined on gold source spans (half-open `[start, end)` char offsets into
  the cleaned document text), never on chunks, so every chunker is scored against the
  same labels (`src/ragbench/eval/spans.py`).
- A chunk hits a gold span of the same document when the overlap is positive and
  `overlap >= f * len(chunk)` **or** `overlap >= f * len(gold span)`, with
  `f = --min-overlap-frac`, default **0.2** (recorded per run as `min_overlap_frac`).
  The OR form follows spec section 5.6: a small chunk wholly inside a large section, or a
  large chunk wholly containing a small span, both count.
- First-hit crediting (used by every metric): each gold span is credited once, to the
  first ranked chunk that hits it; a later chunk is relevant only if it hits a span not
  yet credited. Fine-grained chunkers therefore cannot inflate nDCG by returning several
  slices of one section. All 157 verified questions have exactly one gold span.
- Metrics per question, averaged over the split: Recall@{1,3,5,10} (any relevant chunk
  in the top k), MRR@10, and binary-gain nDCG@10 with an ideal DCG of
  `min(n_gold_spans, 10)` leading hits (`src/ragbench/eval/retrieval_metrics.py`).

## Retrieval index parameters

- HNSW parameters, equalized across pgvector and Qdrant: `m=16`, `ef_construction=64`,
  query-time `hnsw.ef_search=100` (pgvector, Task 2.5; Qdrant parity Task 2.6)
- control_bm25 rationale: separates backend effect from lexical-scorer effect (Phase 2)
- Exact-search sanity check (`ann_sanity_recall` in every dense/hybrid run JSON): for up
  to 50 seeded split questions, the store's HNSW top-`final_k` is compared with
  brute-force cosine over every chunk vector. An ANN result counts as correct when its
  exact cosine is at least the exact k-th best (minus 1e-6), so ties between identical
  chunk texts are not misses. Expected >= 0.98.

## Statistics

- Percentile bootstrap over questions: `numpy.random.default_rng(seed)` draws an index
  matrix `(n_resamples, n)`, each row's mean is taken, and the 2.5 / 97.5 percentiles
  (linear interpolation) form the 95% CI. Defaults: **1000 resamples, seed 42** (the
  config seed); both are recorded per run (`src/ragbench/eval/stats.py`).
- Config-vs-baseline deltas use a paired bootstrap: the same procedure over
  per-question differences on identical question sets. A difference whose paired CI
  contains 0 is not claimed.
- At a metric's ceiling (every question scores 1.0) the CI collapses to `[1, 1]`. That
  shows saturation, not certainty.

## Latency and resources

- Query latency: single request at a time, every split question in order, `--passes`
  passes (default 3, minimum 3); pass 0 is warm-up and discarded; p50/p95 per stage
  (`embed_query, dense, sparse, fusion, rerank, total`; stages a mode skips report 0)
  come from the remaining passes (`src/ragbench/eval/latency.py`). Hardware:
  `results/hardware.json` (referenced by every run).
- Query embeddings are always computed by the model, never served from the disk cache,
  so `embed_query` measures real inference (only document embeddings are cached).
- Indexing: wall time, `embed_seconds`, chunks/sec (`chunks / embed_seconds`), store
  `size_bytes` (pgvector `pg_total_relation_size`; Qdrant per Task 2.6; control_bm25 an
  in-process approximation), and peak process RSS sampled every 50 ms with psutil.
  **Caveat:** document embeddings are disk-cached, so `embed_seconds` and chunks/sec
  reflect the cache state at index time (and include the lazy model load on the first
  embed). Throughput is only comparable across configs when measured cold. See
  OPEN_QUESTIONS.

## Generation judging

- Sweep (spec section 5.8, `configs/grid.yaml`): stage A chunker x embedder (dense,
  pgvector, 15 configs) -> B retrieval mode incl. BM25 control -> C reranker (none,
  cross-encoder top_n 20/50) -> D backend. Each stage expands the parent stage's top 3 by
  **dev nDCG@10** (tiebreak MRR@10, Recall@5, config_id). The top 5 configs overall on dev
  are the only ones evaluated on test (`ragbench finalize`); test never informs selection.
- Device: every sweep and finalist run executes on the same GPU (RTX 3050 Laptop, 4 GB,
  torch `2.15.0+cu132`, `results/hardware.json`): embeddings, query encoding and the
  cross-encoder. Cold embedding throughput per embedder (cache bypassed) is measured
  separately by `ragbench bench-embed` -> `results/embedder_throughput.json`.
- Generation: prompt `gen-v1` (`src/ragbench/generation/prompt.py`), the config's top
  `context_k` (5) chunks, temperature 0, generator `LLM_MODEL`.
- Judging: `ragas==0.4.3` collections metrics (faithfulness, answer relevancy with
  strictness 3 and local `BAAI/bge-small-en-v1.5` query embeddings, reference-based context
  precision, context recall). Judge temperature 0 and `top_p=1.0` (ragas defaults to 0.1,
  which Mistral models behind the `rag` combo reject under greedy sampling). No `seed` is
  sent because some routed providers reject it as an unknown parameter. Instructor JSON
  mode. Per-metric judge error counts are logged after each finalist. Run on a fixed seeded subset of 100 test questions per
  finalist. Failed or NaN judge outputs are stored as missing and excluded per metric;
  `n_valid` is reported. Every generation block records the generator and judge models,
  prompt version, ragas version and embedder.
- Human audit: `docs/judge_spotcheck.md` holds 20 seeded judged items for the human to
  check (spec section 5.7); the agent never fills in verdicts.

## Regression gate

- Baseline provenance, threshold, degraded-config failure evidence: (Phase 5)

## Resolved versions

- Python and package pins: `pyproject.toml`, `uv.lock` (recorded here at release)
- pgvector extension: `0.8.7` in image `pgvector/pgvector:0.8.7-pg16-trixie`
  (verified via `SELECT extversion FROM pg_extension WHERE extname='vector'`, Task 2.5)
- Model revisions: (Phase 2)
