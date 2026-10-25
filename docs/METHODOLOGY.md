# Methodology

> Stub: sections are filled as phases land. Content is written only from
> observed results and verified documentation — never guessed.

## Corpus

- Pinned source, tag/SHA, license, fetch date: see `corpus.lock`.
- Cleaning rules and section parsing: (Phase 1)

## Gold set

- Candidate generation prompt + filters, lexical-overlap threshold used: (Phase 1)
- Human verification protocol, verified count, dev/test split: (Phase 1)

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
