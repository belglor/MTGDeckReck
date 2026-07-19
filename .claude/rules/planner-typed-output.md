---
paths:
  - "src/mtg_rag/plan/**"
---

Planner output is validated against the typed `[{query_text, purpose}]` schema — never parse free-form model prose into queries, and let a validation failure raise rather than degrade.
Why: [ADR 0004](../../docs/adr/0004-planner-typed-query-schema.md).
