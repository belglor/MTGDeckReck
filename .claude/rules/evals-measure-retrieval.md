---
paths:
  - "src/mtg_rag/evals/**"
---

Evals assert recall on the retrieved candidate pool — never on curated output, its ordering, or its length, and only for queries whose expected cards are mechanically determined.
Why: [ADR 0006](../../docs/adr/0006-eval-measures-retrieval-recall.md).
