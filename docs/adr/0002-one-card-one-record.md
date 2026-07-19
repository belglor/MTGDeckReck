---
status: "accepted"
date: 2026-07-19
---

# One card is one record; no document chunking

## Context and Problem Statement

RAG systems conventionally chunk source documents so that each embedded unit is small and topically focused. A Magic card is already small — a name, a type line, and a few lines of oracle text — but some cards carry long rules text, and each card also drags along rulings, flavor text, and multiple printings. What should the atomic unit of retrieval be?

## Considered Options

- One record per card, embedding name + type line + oracle text as a single text channel
- Chunk cards with long oracle text into multiple records
- One record per card *face*, so modal and double-faced cards contribute several records
- Split rulings and flavor text into sibling records that link back to the card

## Decision Outcome

Chosen option: "One record per card", because the card *is* the unit the user receives. Every other option retrieves a fragment that then has to be resolved back to a card before it means anything, and introduces a dedupe problem — the same card surfacing three times from three chunks — in exchange for granularity nothing downstream can use. A recommender's answer is a list of cards, so the thing we search over should be cards.

Chunking exists to stop a single embedding from having to represent too many unrelated ideas. Oracle text is short enough and coherent enough that this pressure does not apply: the longest cards are still a paragraph, and their abilities are thematically related by design.

### Consequences

- Good, because a retrieval hit is directly an answer — no fragment-to-card resolution step, no cross-chunk deduplication
- Good, because recall metrics are countable in the same units the user thinks in ("did Urza's Mine come back?"), which is what makes [ADR 0006](0006-eval-measures-retrieval-recall.md) straightforward to write
- Good, because the corpus stays at roughly 30k records, small enough that a local vector store needs no infrastructure
- Bad, because a card whose abilities span several distinct themes gets one averaged embedding, so a very long card is slightly harder to retrieve on any *one* of its themes
- Bad, because rulings and flavor text are excluded from the embedded channel rather than searchable in their own right — a query phrased in flavor terms has only the oracle text to match against
