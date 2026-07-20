---
paths:
  - "src/mtg_rag/evals/**"
---

Evals assert recall on the retrieved candidate pool — never on curated output, its ordering, or its length, and only for queries whose expected cards are mechanically determined.
Why: [ADR 0006](../../docs/adr/0006-eval-measures-retrieval-recall.md).

Compare recall@k only within a fixed embedding configuration — a change to the model, channel set, or dimension starts a new baseline; record the new number, never diff it against the old one.
Why: [ADR 0011](../../docs/adr/0011-evaluation-scope-and-baseline-semantics.md).
