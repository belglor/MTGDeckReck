---
paths:
  - "src/mtg_rag/embed/**"
  - "src/mtg_rag/store/**"
  - "src/mtg_rag/retrieve/**"
---

Key every vector by `oracle_id` — never by Scryfall's `id` (a printing identifier that changes when the bulk file re-picks a card's printing) or by card name. Store no card metadata alongside vectors: filters are evaluated against the corpus parquet to produce an id allowlist, the search is constrained to that allowlist, and the text shown to the LLM is read back from the parquet. A copy of legality, colour identity, or card text living next to a vector is a second source of truth and must not be created.
Why: [ADR 0010](../../docs/adr/0010-oracle-id-identity-key.md).
