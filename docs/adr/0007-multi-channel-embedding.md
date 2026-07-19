---
status: "accepted"
date: 2026-07-19
---

# Embed card properties as separate semantic channels; structured facts stay filters

## Context and Problem Statement

A card record ([ADR 0002](0002-one-card-one-record.md)) has many properties: oracle text, flavor text, type line, rulings, artist, and a pile of structured facts — legality, color identity, price, rarity, mana value, set. Concatenating name, type line, and oracle text into one embedding is the obvious starting point, but it gives a flavor-phrased request nothing but rules text to match against, and this recommender's premise is flavor and theme over meta. "A spooky graveyard deck" lives in flavor text far more than in oracle text. Which properties get embedded, and as one vector or several?

## Considered Options

- A single concatenated channel: name + type line + oracle text
- Separate embedding channels for the semantic text properties, fused at query time
- Embed every available Scryfall property, structured fields included, and let fusion sort it out
- A single channel plus richer structured filters, with no additional embedded text

## Decision Outcome

Chosen option: "Separate channels for semantic text", with three to start: **oracle text** (card name folded in, since the name is a rules-text-adjacent identifier), **flavor text**, and **type line** including creature types. Each is embedded independently and produces its own ranking; the rankings are combined per [ADR 0008](0008-rrf-fusion-not-raw-scores.md).

Separate channels beat one concatenated channel because the properties are different registers of language. Oracle text is terse rules prose; flavor text is evocative narrative; a type line is a controlled vocabulary. Averaging them into one vector lets the longest one dominate and blurs the signal that makes each useful — and it means a query written in one register has to match a vector mostly composed of the others.

The rejected option is "embed everything". Scryfall's properties split cleanly in two, and the split follows the line [ADR 0001](0001-legality-color-as-filters-not-prompts.md) already drew:

- **Semantic text** — oracle, flavor, type line, rulings — is prose with meaning to approximate. It gets a channel.
- **Structured facts** — legality, color identity, price, rarity, mana value, set, release date — have exact answers. They stay filters and sort keys.

Embedding a price asks a vector to approximate `≤ $5`, which a `WHERE` clause answers exactly and a nearest-neighbour search answers only by coincidence. Embedding legality would actively contradict 0001. Structured data does not become more useful for being made fuzzy.

Rulings and artist are deliberately left out for now. Both are plausible channels — rulings disambiguate named mechanics, artist correlates with visual mood — but neither is needed for the first end-to-end flow, and each additional channel has a measurable cost. Adding one later is a small change.

### Consequences

- Good, because a flavor-phrased request finally has a surface built for it, which is the kind of request this recommender exists to serve
- Good, because each channel embeds one homogeneous register, so no channel's signal is diluted by text of a different kind
- Good, because channels are independently measurable and independently improvable — a bad flavor channel can be diagnosed and replaced without touching oracle retrieval
- Bad, because ingestion and storage cost scale with channel count: three passes over ~30k cards instead of one
- Bad, because query cost multiplies against a planner query count that is already variable ([ADR 0004](0004-planner-typed-query-schema.md)) — eight planned queries across three channels is twenty-four vector searches for one request
- Bad, because many cards have no flavor text, so that channel indexes a smaller corpus than the others; its top-k is not evidence of a good match and must not be read as one
- Bad, because adding, removing, or re-embedding a channel invalidates the recall baseline in [ADR 0006](0006-eval-measures-retrieval-recall.md) — the golden set must be re-run and the new number recorded as the baseline, not compared against the old one
