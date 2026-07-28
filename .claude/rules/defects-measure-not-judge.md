---
paths:
  - "src/mtg_rag/defects/**"
---

Measure only defects a string or set operation can prove — a repeat, a copied example, a rationale contradicting the corpus. Never score whether a query is a good search or a rationale a persuasive argument: that is taste, it needs human judgment, and an LLM judge would make the instrument as variable as the thing measured.
Why: [ADR 0026](../../docs/adr/0026-measure-output-defects-not-quality.md), [ADR 0006](../../docs/adr/0006-eval-measures-retrieval-recall.md).

Report, never gate. No thresholds, no pass/fail, no verdict in the output, and the CLI always exits 0 — `role_count` in particular is printed precisely because [ADR 0006](../../docs/adr/0006-eval-measures-retrieval-recall.md) refuses to say what a right answer would be. Stays out of `just check` and CI, as `just eval` does.
Why: [ADR 0020](../../docs/adr/0020-eval-case-is-a-corpus-predicate.md).

Measure a proposed check against the real corpus before building it. Two have failed that way already: flagging query terms absent from the corpus as invented vocabulary (real hallucinations pass, `urzatron` fails), and reading any type word in a rationale as a claim about the card (44.8% of commander-legal cards name a type they are not, in their own text). Both read as mechanical and survived review.
Why: [ADR 0026](../../docs/adr/0026-measure-output-defects-not-quality.md).

A run that failed is recorded, never dropped or raised — an instrument that stops at the first bad output cannot measure how often output is bad, and one that silently drops failures looks cleaner the worse the model gets. Numbers are comparable only within a fixed theme set, model and constraint set; changing any of them starts a new baseline rather than continuing one.
