# Methodology

> Content is written only from observed results and verified documentation, never
> guessed. Numbers quoted here come from files in `results/`, `data/goldset/` and
> `eval/`.

## Corpus

- Pinned source, tag/SHA, license, fetch date: see `corpus.lock`.
- Corpus: FastAPI documentation (`docs/en/docs`, 156 markdown files), tag `0.162.0`, MIT.
- Cleaning (`ragbench/corpus/clean.py`): strip YAML front matter, normalize line endings,
  collapse runs of blank lines. Fenced code blocks are kept verbatim (they are tracked by a
  fence mask so headings inside code are not parsed as sections).
- Sections (`ragbench/corpus/sections.py`): split at markdown headings outside code fences;
  each section has a slugged id (`doc_id#slug`), its heading path and character offsets
  into the cleaned document text. 2,199 sections in total.

## Gold set

- Sampling: 506 sections (stratified by document and length; at least 80 tokens; changelog
  and code-dominated sections excluded).
- Generation: one question plus a reference answer per section from the `rag` model.
- Filters (thresholds in `data/goldset/generate_stats.json`): heading-copy ratio <= 0.6,
  lexical overlap with the section <= 0.6 (so BM25 is not unfairly favored), embedding dedup
  at cosine 0.92, code-fence ratio <= 0.6, and an answerability judge. 483 candidates
  survived.
- Human verification: every candidate was reviewed in `ragbench goldset review`
  (accept / edit / reject / flag). 157 were accepted, 1 rejected and 18 flagged. Only
  verified questions are used.
- Split: seeded (42), stratified by question type, dev 47 (30%) / test 110 (70%). Dev is used
  only for selection, test only for the final numbers.

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
- control_bm25 rationale: an in-process BM25 (`bm25s`, OR semantics) that is the same for
  every backend. It separates the effect of the backend from the effect of its lexical
  scorer. This is how the pgvector FTS issue was found: pgvector sparse scored nDCG@10
  about 0.04 while control_bm25 scored 0.815 on the same chunks.
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
  mode. Per-metric judge error counts are logged after each finalist. `rag` is an OmniRoute combo that routes each call to one model from a pool (Grok and Mistral models), so the underlying generator and judge model can differ between calls and finalists. Generation-metric differences between finalists of about 0.1 are within run-to-run noise (two finalists with identical contexts differed by 0.06 in faithfulness). Run on a fixed seeded subset of 100 test questions per
  finalist. Failed or NaN judge outputs are stored as missing and excluded per metric;
  `n_valid` is reported. Every generation block records the generator and judge models,
  prompt version, ragas version and embedder.
- Human audit: `docs/judge_spotcheck.md` holds 20 seeded judged items for the human to
  check (spec section 5.7); the agent never fills in verdicts.

## Regression gate

- **What runs:** the CI job `regression-eval` runs `ragbench regression-check` with
  `configs/regression_mini.yaml` on the mini fixture's test split (31 questions). That config
  is recursive_md-256 chunks, `all-MiniLM-L6-v2`, pgvector dense and no reranker, so it is
  CPU-friendly. Results go to `.tmp/regression/`, never to `results/`.
- **Baseline:** `eval/baseline.json`, written by `make update-baseline` (config
  `482487b7aba3`, 2026-11-03): Recall@5 0.871, MRR@10 0.686. It was generated on the local
  GPU; CI runs on CPU. Embeddings agree to float tolerance, so ranks match except for
  near-ties.
- **Thresholds:** `configs/regression_thresholds.json`. The gate fails (exit 1) if Recall@5
  drops more than **0.03** absolute or MRR@10 more than **0.05**.
- **Negative evidence:** `configs/regression_degraded.yaml` is the same config with
  `final_k: 3`. The gate fails as intended:

  ```text
  FAIL recall@5: baseline=0.8710 current=0.7742 drop=+0.0968 (max 0.0300)
  PASS mrr@10: baseline=0.6863 current=0.6559 drop=+0.0304 (max 0.0500)
  regression: 1 metric(s) dropped past threshold
  ```

- **Why not 16-token chunks:** the plan suggested chunk size 16 as the degraded config, but
  it does **not** regress (Recall@5 0.871, unchanged; MRR@10 0.674). Under the span-overlap
  rule, a tiny chunk lying inside a gold section covers 100% of itself, so it counts as a hit.
  Recall at the gold-span level is therefore robust to very small chunks. Retrieval with 16-token
  chunks still finds the right sections. A useful answer would need more context than that, but
  retrieval metrics do not measure it.

## Resolved versions

- Python 3.13; every package is pinned in `pyproject.toml` and `uv.lock`. Every run JSON
  records its key package versions; the finalists ran with numpy 2.5.4, torch
  2.15.0+cu132, sentence-transformers 6.2.0, transformers 5.22.1, qdrant-client 1.20.0,
  pgvector 0.5.1, psycopg 3.3.6, fastembed 0.9.1, bm25s 0.3.16, tiktoken 0.14.0,
  pydantic 2.14.1 and ragas 0.4.3.
- Qdrant server `v1.20.0`.
- pgvector extension: `0.8.7` in image `pgvector/pgvector:0.8.7-pg16-trixie`
  (verified via `SELECT extversion FROM pg_extension WHERE extname='vector'`, Task 2.5)
- Model revisions (Hugging Face commit of the cached snapshot):

  | Model | Revision |
  |---|---|
  | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
  | `intfloat/e5-small-v2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
  | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
  | `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` |
  | `Qdrant/bm25` (fastembed sparse) | `22b8d2af71a76161e18dd432d2cee0eefa66e412` |

- LLM: OmniRoute combo `rag` (a router pool, not one model), prompt `gen-v1`, temperature
  0; see `docs/LIMITATIONS.md`.
