---
status: "accepted"
date: 2026-07-23
---

# Platform availability is the union across a card's printings

## Context and Problem Statement

Retrieval is about to offer a platform filter — paper, MTGO or Arena — alongside format legality and color identity ([ADR 0001](0001-legality-color-as-filters-not-prompts.md)). A user building a paper Commander deck should not be shown cards that exist only on Arena.

Scryfall reports this as `games`, but it reports it **per printing**. Ingestion collapses printings to one record per card ([ADR 0016](0016-ingest-every-printing.md)), and taking `games` from the representative printing gives an answer to the wrong question: it says which media carry *that printing*, not which carry the card.

The gap is not academic. Under the previous corpus, 1,058 cards looked MTGO-only and 1,042 of those were commander-legal — Palinchron, Wyluli Wolf, Weakstone and hundreds more, all ordinary paper cards whose representative printing happened to be an MTGO-only Masters Edition reprint. Sampling 20 of them against Scryfall's API, 20 of 20 had a paper printing. The Arena side was worse in relative terms: 3 of 20 sampled cards whose stored printing lacked `arena` did have an Arena printing.

## Considered Options

- Union `games` across every printing into a card-level `platforms` field
- Filter on the representative printing's `games` as ingested
- Treat `mtgo` as implying paper, since MTGO-only reprint sets are reprints of paper cards
- Keep the field per-printing and have the filter accept a card if any printing matches

## Decision Outcome

Chosen option: "Union across every printing", replacing the `games` column with `platforms`.

Filtering the representative printing's value was rejected outright: it hides 1,042 commander-legal paper cards, and it fails silently, which is the worst combination available.

Treating `mtgo` as implying paper would have patched the paper case without touching the Arena one. A user selecting Arena would still miss thousands of cards, and the rule encodes a coincidence about which sets are MTGO-exclusive rather than a fact about cards.

Filtering across printings at query time is the same computation, moved somewhere it costs more: the corpus stores one row per card ([ADR 0002](0002-one-card-one-record.md)), so there is nowhere for per-printing data to live without tripling the parquet and reopening a decision that is already made.

**The vocabulary is restricted to `paper`, `mtgo` and `arena`.** Scryfall also reports `astral` (10 cards from the 1997 Shandalar PC game) and `sega` (8 Dreamcast promos). Those are curiosities, not places anyone plays. They are dropped at projection, so the 18 cards printed only there end up with an empty `platforms` and are invisible to every platform filter — which is the intended outcome, arrived at deliberately rather than by omission. The cards keep their rows; excluding them belongs to the filter, not to ingestion.

`CardRecord.games` becomes `CardRecord.platforms` rather than the two coexisting. The field means "where this record can be played", which for a printing is its own media and for a merged card is the union — the value follows the record's scope instead of two names competing for one concept.

### Measured outcome

Across the 34,201 real cards in the corpus:

| | before | after |
|---|---|---|
| Commander-legal cards lacking `paper` | 1,042 | 1 |
| Cards carrying `arena` | 12,129 | 15,962 |
| MTGO-exclusive real cards | 1,058 | 2 |
| Arena-exclusive real cards | 989 | 984 |
| Cards with no platform at all | 0 | 18 |

The one remaining commander-legal card without paper is `"Name Sticker" Goblin`, which has a single printing, in the paper set Unfinity, that Scryfall itself reports as MTGO-only. That is upstream being wrong, and one card is a reasonable amount of upstream to be wrong about.

The 18 cards with no platform are exactly the `astral` and `sega` sets described above.

### Consequences

- Good, because the filter [ADR 0001](0001-legality-color-as-filters-not-prompts.md) requires can now be built on a column that answers the question it asks
- Good, because it costs no re-embed and no re-index: the corpus parquet is the single source of truth for filters ([ADR 0010](0010-oracle-id-identity-key.md)), so rewriting it is the whole job
- Good, because a card's platforms no longer change when Scryfall re-picks its representative printing, which it does on every build
- Bad, because `platforms` is now the second field, after flavor text, that ignores the representative-printing rule — the rule has two exceptions and any third deserves scrutiny
- Bad, because dropping `astral` and `sega` at projection means the corpus can no longer answer a question about them at all, however unlikely that question is
- Bad, because the corpus inherits upstream errors with no way to detect them: `"Name Sticker" Goblin` is wrong in the parquet because it is wrong at Scryfall
