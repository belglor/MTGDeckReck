---
status: "accepted"
date: 2026-07-22
---

# Ingest every printing and collapse it with an explicit representative-printing rule

## Context and Problem Statement

Ingestion read Scryfall's `oracle_cards` bulk file, which ships one object per card by picking a single printing to stand for it — its documentation calls this "an attempt to select the most up-to-date recognizable version". But several of the fields that projection carries belong to a *printing*, not to a card: `games`, `rarity`, `flavor_text`, `set`, `released_at`, and prices. The corpus therefore inherited whichever printing Scryfall chose, without recording that a choice had been made, and [ADR 0010](0010-oracle-id-identity-key.md) already notes that the choice is re-made on every build.

That is fine until something needs to know a fact about the *card*. Platform availability is the first such question and the reason this came up: a filter asking "can I play this in paper" cannot be answered from one printing, because Scryfall may have picked an MTGO-only reprint of an ordinary paper card. The same problem shapes flavor text, where a card may carry dozens of different passages across its printings and the corpus stores exactly one.

Which bulk file should ingestion read, and if it reads every printing, what decides the contents of the one row per card that [ADR 0002](0002-one-card-one-record.md) requires?

## Considered Options

- Keep `oracle_cards`, and join a second pass over `default_cards` for the per-printing facts that need it
- Read `default_cards` and collapse printings with an explicit rule of our own
- Read `default_cards` and keep one row per printing, collapsing at query time
- Keep `oracle_cards` and accept that per-printing fields answer a different question than the one asked

## Decision Outcome

Chosen option: "Read `default_cards` and collapse printings with an explicit rule", because it has one source, one download, and one timestamp, and because measurement showed the collapse to be far smaller than expected.

Across all 116,138 printing objects, **`oracle_text`, `type_line` and legality vary on exactly zero cards** — Scryfall applies current oracle text to every printing, and legality is a property of the card rather than the cardboard. So the aggregation is not a general merge problem. Only `rarity` (3,316 cards), `flavor_text` (3,550 carry more than one), and the set, release date, `games` and prices differ at all.

The rule is a **representative printing**: the most recent real printing, tie-broken on set code for determinism. It supplies every single-valued field, so a row describes one physical card rather than a composite — `price_usd` belongs to the same object as the `set_code` and `rarity` beside it.

Two refinements were forced by measurement rather than anticipated.

**Flavor text is taken from the most recent printing that has any**, not from the representative printing. Taking it from the representative would have stripped flavor text from 1,804 cards, because a card's newest printing is so often a Commander-deck or promo reprint carrying none, and the flavor channel is already the sparsest of the three ([ADR 0007](0007-multi-channel-embedding.md)).

**A printing that is not itself a real card may never represent one** ([ADR 0013](0013-structural-card-predicate.md)). Ranking on date alone gave Tundra its 30th Anniversary Edition printing, whose `set_type` is `memorabilia` — so the structural predicate then discarded Tundra as a non-card. This removed **302 real cards**, Mox Jet and the World Championship reprints among them, and was caught by the existing invariant test asserting that the predicate drops no commander-legal card. `is_real` was added beside `is_real_card` so ingestion can apply the same rule to one printing's values before a frame exists.

Only English printings are ingested. `default_cards` falls back to a foreign printing for cards that exist in no English one, and such a printing would otherwise win the contest and seed an embedding channel with a language nothing else in the corpus uses.

Reading every printing and collapsing at query time was rejected outright: it contradicts [ADR 0002](0002-one-card-one-record.md), triples the parquet, and moves work into the request path to no benefit, since the collapse depends on nothing the request supplies.

### Measured outcome

| | before | after |
|---|---|---|
| Rows | 38,312 | 38,320 |
| Real cards ([ADR 0013](0013-structural-card-predicate.md)) | 34,184 | 34,193 |
| Commander-legal | 31,622 | 31,622 |
| Cards with flavor text | 20,248 | 20,889 |
| Download | 22.8 MB gz | 72.6 MB gz |

On the 34,176 real cards common to both, `oracle_text`, `type_line`, `name`, `mana_cost`, `cmc`, `keywords` and legality are byte-identical — the invariance claim, asserted rather than trusted. Flavor text was gained by 636 cards, changed on 1,079, and **lost by none**. `rarity` moved on 969. Eight cards left the corpus: the Japanese-only Sega Dreamcast promos, which have no English printing. Seventeen arrived from sets `oracle_cards` had not yet picked up.

### Consequences

- Good, because facts that differ between printings can now be answered at card level at all — which is what [ADR 0001](0001-legality-color-as-filters-not-prompts.md)'s platform filter needs
- Good, because the rule deciding a row's contents is ours, written down, and stable, instead of an upstream heuristic that is re-run on every build
- Good, because a row is internally coherent: set, rarity, release date and price describe one physical printing
- Good, because flavor coverage improved by 641 cards without a single card losing text
- Bad, because ingestion now downloads 72.6 MB instead of 22.8 MB and parses 113,494 objects instead of 38,312
- Bad, because the flavor channel's content changed for roughly 1,700 cards, so the retrieval baseline resets rather than continues ([ADR 0011](0011-evaluation-scope-and-baseline-semantics.md))
- Bad, because "most recent" is a rule with no claim to being the *best* printing — it is defensible and deterministic, not optimal, and a card whose newest printing has an unusual rarity now reports that rarity
- Bad, because the structural predicate now has two implementations, one per shape, kept in step by a test rather than by construction
