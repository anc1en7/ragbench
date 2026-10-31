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

## Relevance scoring

- Span-overlap rule and minimum-fraction setting: (Phase 3)
- nDCG first-hit-per-gold-span handling: (Phase 3)

## Retrieval index parameters

- HNSW parameters, equalized across pgvector and Qdrant: (Phase 2)
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
- Model revisions: (Phase 2)
