# Limitations

> Stub: filled honestly as the system is built (spec rule 9). This project
> does not claim state-of-the-art results.

1. **Judge-model bias.** Generation and judging use the same model (`LLM_MODEL=rag`, an
   OmniRoute combo pooling Ministral 14B / Grok) unless `LLM_JUDGE_MODEL` is set, so
   self-preference bias is possible. The combo may route requests to different
   underlying models, which adds judge variance. RAGAS runs on 100 test questions per
   finalist only. A 20-item human spot-check (`docs/judge_spotcheck.md`) is the only
   audit of judge quality.
2. **Synthetic questions with human verification.** Questions are LLM-written
   and human-verified, not sourced from real user traffic.
3. **Single corpus.** FastAPI documentation only; results may not transfer to
   other domains or document styles.
4. **Single hardware profile.** Latency numbers reflect one machine (see
   `results/hardware.json`); they are not portable across hardware.
5. **ANN approximation.** Retrieval uses HNSW; exact-search agreement is
   reported once as a sanity check but individual queries may differ from
   brute force.
6. **Small embedding models on CPU-class hardware.** Model selection follows
   the free/local-first constraint; larger embedders were not evaluated.
7. **Laptop GPU latency.** The sweep ran locally on an RTX 3050 Laptop GPU (4 GB) with
   a 12-thread i5-12450H. Absolute latencies (especially the cross-encoder) reflect that
   machine; a CPU-only deployment would be much slower. Relative comparisons hold only
   within this one device.
8. **Staged, not exhaustive, search.** Later stages only expand the top 3 configs of the
   previous stage (spec section 5.8), so factor interactions outside those paths are
   untested.
9. **Generator and judge model not pinned.** `rag` is a router pool, so faithfulness and
   answer relevancy vary by about +/-0.1 between runs. Finalists that received identical
   contexts differ by 0.06 in faithfulness, so RAGAS differences between finalists are not
   evidence that one pipeline is better. A re-run with one pinned model was not done
   (Checkpoint 4 decision, 2026-11-03).
10. **High refusal rate from the generator.** 33-56% of answers say the context does not
    contain the answer, even though context recall is about 0.95. This lowers faithfulness
    and answer relevancy. The `gen-v1` prompt was not revised.
11. **pgvector lexical search uses AND semantics.** `websearch_to_tsquery` requires every
    term to match, so pgvector sparse returns hits for only 3/47 dev questions and pgvector
    `hybrid` equals pgvector `dense`. Qdrant's BM25 path and the in-process `control_bm25`
    are unaffected. The fix (OR the lexemes) was not applied.
