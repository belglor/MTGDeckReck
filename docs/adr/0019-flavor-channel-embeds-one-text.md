---
status: "accepted"
date: 2026-07-24
---

# The flavor channel embeds one text per card: the most recent printing that has any

## Context and Problem Statement

A card can carry different flavor text on every printing — 3,479 cards have more than one distinct text, and one has 54. Ingestion already picks a single one for the corpus: the most recent printing that actually has flavor text ([ADR 0016](0016-ingest-every-printing.md)). The flavor channel then embeds that.

That was a decision about which text to *store*, not about what the channel should *search*. The question left open is whether one vector per card is the right representation, or whether a card's other flavor texts should reach retrieval too.

## Considered Options

- Embed the one stored text — the most recent printing that has flavor text
- Concatenate every distinct flavor text into one document and embed that
- Embed each flavor text separately and average the vectors into one
- Embed each flavor text as its own vector, several per card
- Add a second channel for the non-primary variants

## Decision Outcome

Chosen option: "Embed the one stored text", because it is already computed, costs nothing, and every alternative is a bet on retrieval quality that this project currently has no way to settle.

That is the whole rationale, and it is deliberately modest. [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md) makes the golden set the instrument for comparing retrieval configurations, and it does not exist yet. Each alternative below is arguable on paper and none is measurable today, so picking one would be choosing on taste and paying a re-embed to do it. The priority is a working end-to-end product; retrieval quality is tuned once there is something to tune against.

**Concatenation** was rejected on a concrete limit rather than a judgement: `MAX_SEQ_LENGTH` is 512 tokens, and a card with dozens of flavor texts exceeds it, so the tail would be silently truncated. Choosing deliberately beats truncating by accident.

**Averaging the per-text vectors** is the option most worth naming, because it is the one a reader reaches for and it is not obviously wrong. It was rejected for two reasons that hold without an eval. First, it costs exactly what several-vectors-per-card costs — every text must be embedded either way, 27,455 against today's 21,072 — and then discards the per-text information the multi-vector option keeps; it pays the higher price for the smaller result. Second, averaging assumes the parts are facets of one coherent thing, the way chunks of a document are. A card's flavor texts are not that: they are alternative renderings written independently per printing, often in a different voice about a different scene, so their mean is a point that corresponds to no printed text. Sampling confirmed the divergence is real and large; the figures live in issue #47, since they argue about how much is being *given up*, not about what to do now.

**Several vectors per card** is the most promising alternative and is deferred rather than dismissed. It matches the question retrieval actually asks — whether *any* of a card's flavor fits the request — and it keeps every text. Its cost is that [ADR 0010](0010-oracle-id-identity-key.md) keys every vector by `oracle_id` and assumes those are unique within a collection, so it needs composite ids and a fold-back step in fusion. That is a contract worth reopening on evidence, not on preference.

**A second channel** would change the channel set, which [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md) treats as a new recall baseline, for a benefit nothing has measured.

### Consequences

- Good, because it ships today: the text is already in the corpus and the channel already embeds it, so this decision costs no code and no re-embed
- Good, because one vector per card keeps [ADR 0010](0010-oracle-id-identity-key.md)'s identity contract intact — `oracle_id` stays unique within every collection
- Good, because it defers a quality question until there is an instrument to answer it, rather than guessing and calling the guess a decision
- Bad, because the flavor a card is searchable by is decided by print date, which has nothing to do with which text best represents the card
- Bad, because the discarded texts are not near-duplicates — they are semantically distinct, so this is real information loss rather than deduplication
- Bad, because the loss is uneven: it falls entirely on the 3,479 cards with several texts and not at all on the other 17,593, so the channel represents some cards better than others for reasons unrelated to them
