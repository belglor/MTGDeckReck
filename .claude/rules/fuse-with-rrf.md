---
paths:
  - "src/mtg_rag/retrieve/**"
---

Combine channel and per-query rankings with Reciprocal Rank Fusion over ordinal positions — never average, sum, or take the max of raw similarity scores across channels, which are not commensurable.
Why: [ADR 0008](../../docs/adr/0008-rrf-fusion-not-raw-scores.md).
