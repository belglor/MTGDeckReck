---
paths:
  - "src/mtg_rag/embed/**"
---

Only semantic text gets an embedding channel — oracle text, flavor text, type line. Never embed structured facts (legality, color identity, price, rarity, mana value, set); they are filters and sort keys, and a vector only approximates what a comparison answers exactly.
Why: [ADR 0007](../../docs/adr/0007-multi-channel-embedding.md).
