# Limitations

> Stub: filled honestly as the system is built (spec rule 9). This project
> does not claim state-of-the-art results.

1. **Judge-model bias.** Generation and judging use the same configured
   endpoint; self-preference bias is possible. (Quantified in Phase 4.)
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
7. **CI-generated sweep.** The staged sweep runs on shared GitHub Actions
   runners; absolute latencies from that environment reflect the runner, not
   consumer hardware, and are labeled accordingly in results.
