---
paths:
  - "src/mtg_rag/ingest/**"
  - "src/mtg_rag/embed/**"
---

One card is one record — never chunk oracle text, and never split a card across multiple records (by face, by ruling, or otherwise). Several vectors per record is expected ([ADR 0007](../../docs/adr/0007-multi-channel-embedding.md)); several records per card is not.
Why: [ADR 0002](../../docs/adr/0002-one-card-one-record.md).
