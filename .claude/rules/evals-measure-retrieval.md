---
paths:
  - "src/mtg_rag/evals/**"
---

Measure the retrieved candidate pool — never curated output, its ordering, or its length — and only for queries whose answer is mechanically determined.
Why: [ADR 0006](../../docs/adr/0006-eval-measures-retrieval-recall.md).

An eval case's expected set is a corpus predicate, never a list of card names. Report lift: the predicate's share of the pool divided by its base rate in the **constrained** corpus, `frame.filter(constraint_expr(constraints, frame))`. Never divide by the whole corpus — the wrong denominator produces plausible-looking nonsense rather than an error. Constraint interaction is lift retention between constraint sets.
Why: [ADR 0020](../../docs/adr/0020-eval-case-is-a-corpus-predicate.md).

The eval reports and never fails a run — no thresholds, no gates. Pool invariants (legality, color identity, platform) are not eval cases; they belong to the unit tests that already assert them in `tests/test_retrieve_pool.py`, where a failure blocks a merge.
Why: [ADR 0020](../../docs/adr/0020-eval-case-is-a-corpus-predicate.md), [ADR 0011](../../docs/adr/0011-evaluation-scope-and-baseline-semantics.md).

Compare numbers only within a fixed embedding configuration — a change to the model, channel set, or dimension starts a new baseline; record the new number, never diff it against the old one.
Why: [ADR 0011](../../docs/adr/0011-evaluation-scope-and-baseline-semantics.md).
