---
status: "accepted"
date: 2026-07-19
---

# One card is one record; no document chunking

## Context and Problem Statement

RAG systems conventionally chunk source documents so that each embedded unit is small and topically focused. A Magic card is already small — a name, a type line, and a few lines of oracle text — but some cards carry long rules text, some have multiple faces, and each card drags along rulings, flavor text, and multiple printings. What is the atomic unit of retrieval?

This decides the *unit* only. What text gets embedded for a given card is a separate question, settled in [ADR 0007](0007-multi-channel-embedding.md).

## Considered Options

- One record per card
- Chunk cards with long oracle text into multiple records
- One record per card *face*, so modal and double-faced cards contribute several records
- Split rulings and flavor text into sibling records that link back to the card

## Decision Outcome

Chosen option: "One record per card", because the card *is* the unit the user receives. Every other option retrieves a fragment that then has to be resolved back to a card before it means anything, and introduces a dedupe problem — the same card surfacing three times from three chunks — in exchange for granularity nothing downstream can use. A recommender's answer is a list of cards, so the thing we rank should be cards.

Chunking exists to stop a single embedding from having to represent too many unrelated ideas. Oracle text is short enough and coherent enough that this pressure does not apply: the longest cards are still a paragraph, and their abilities are thematically related by design.

Note that "one record" does not mean "one vector". A record may carry several embeddings of different card properties, fused into a single ranking at query time ([ADR 0007](0007-multi-channel-embedding.md), [ADR 0008](0008-rrf-fusion-not-raw-scores.md)). That is categorically different from chunking: multiple vectors still resolve to one card, whereas chunks fragment the answer unit itself.

### Consequences

- Good, because a retrieval hit is directly an answer — no fragment-to-card resolution step, no cross-chunk deduplication
- Good, because recall is countable in the same units the user thinks in ("did Urza's Mine come back?"), which is what makes [ADR 0006](0006-eval-measures-retrieval-recall.md) straightforward to write
- Good, because the corpus stays at roughly 30k records, small enough that a local vector store needs no infrastructure
- Bad, because a modal or double-faced card is one record covering both faces, so a query matching only the back face competes against text from a front face it has nothing to do with
- Bad, because a card whose abilities span several distinct themes gets one embedding per channel rather than one per theme, making it slightly harder to retrieve on any single theme
