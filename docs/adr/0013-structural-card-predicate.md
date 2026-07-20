---
status: "accepted"
date: 2026-07-20
---

# Exclude only structural non-cards from the index

## Context and Problem Statement

Ingestion keeps every row Scryfall ships ([ADR 0009](0009-parquet-card-corpus.md)) — the `oracle_cards` dump includes tokens, emblems, art-series prints, Planechase planes, Archenemy schemes, and vanguard avatars alongside real cards. `embed/` should not spend vectors on things a player never puts in a deck, but the filter that removes them must not quietly become a legality or format filter, because those are deterministic retrieval-time concerns ([ADR 0001](0001-legality-color-as-filters-not-prompts.md)) and the parquet is meant to own them. What defines "a card worth embedding", and where does that line sit relative to legality?

## Considered Options

- A structural predicate keyed on `layout` and `set_type` only, shared by `embed/` and `retrieve/`
- Reuse the exploration notebook's candidate filter: commander-legal AND paper
- Filter non-cards out at ingest, so the parquet only ever holds real cards

## Decision Outcome

Chosen option: "A structural predicate on `layout` and `set_type` only", because it removes exactly the objects that are not game cards while leaving every policy question — legality, format, `games`, colour — to the retrieval-time filters that already own it.

`is_real_card` excludes these **layouts**: `art_series`, `token`, `double_faced_token`, `emblem`, `vanguard`, `scheme`, `planar`, `augment`, `host`; and these **set types**: `memorabilia`, `token`, `minigame`. Everything else is embedded. Measured against the current corpus:

| | rows |
|---|---|
| corpus | 38,312 |
| after `is_real_card` | **34,184** |
| commander-legal before | 31,622 |
| commander-legal after | **31,622** (zero lost) |

The invariant that matters is the last row: the predicate drops 4,128 rows and **not one of them is commander-legal**. It removes non-cards, never playable cards.

Three calls inside the predicate are non-obvious, and each has a plausible wrong answer:

- **Planes, schemes, and vanguards are excluded by `layout`, never by `set_type`.** `set_type == "planechase"` holds 94 commander-legal cards — ordinary cards printed *in* Planechase products — and a set-type rule would destroy them. The layout rule removes exactly the oversized variant-format objects (the planes themselves) and keeps the normal cards that shipped alongside them. Same shape for Archenemy.
- **Un-cards stay in.** `set_type == "funny"` holds 174 commander-legal cards. Silver-border legality is a legality property, and [ADR 0001](0001-legality-color-as-filters-not-prompts.md) puts legality at retrieval time; structurally an Un-card is a card. Excluding them would bake a legality decision into the index — the exact thing this predicate must not do.
- **Alchemy and other digital-only cards stay in.** They are structurally cards; that they are not paper is what the `games` retrieval-time filter is for, not the index predicate.

The set-type rule is deliberately small and worth flagging as such: layout exclusion alone already keeps 34,256 rows, and the three excluded set types remove only **72** more — the `layout: normal` non-cards that nothing else catches (Face the Hydra, Battle the Horde, the Theros Hero's Path cards, Celebration cards). It earns its place, but no future reader should mistake it for load-bearing.

The rejected options both fold policy into the index. The notebook's "commander-legal AND paper" candidate filter is convenient — it happens to drop every token and emblem as a side effect — but it bakes commander legality and paper-ness into *what got embedded*, so adding a second format later would mean re-embedding rather than changing a filter. Filtering at ingest is worse still: it makes the parquet lossy, contradicts the "keep every row Scryfall ships" contract ([ADR 0009](0009-parquet-card-corpus.md)), and makes questions like "which tokens does this deck make" unanswerable without a re-ingest. The structural predicate keeps the parquet complete and shared between `embed/` and `retrieve/` as one definition of "card".

### Consequences

- Good, because the index holds only real cards, and the same predicate defines "card" for both `embed/` and `retrieve/` — one source of truth, no drift
- Good, because legality, format, `games`, and colour stay retrieval-time filters ([ADR 0001](0001-legality-color-as-filters-not-prompts.md)); a legality change never touches the index, only the parquet ([ADR 0010](0010-oracle-id-identity-key.md))
- Good, because the predicate is `layout`/`set_type` only — cheap, deterministic, and trivially re-checkable against the parquet
- Bad, because the exclusion lists are hand-curated against Scryfall's current vocabulary; a genuinely new `layout` or `set_type` would be embedded by default until someone notices, which is the safe direction to fail but still a manual watch
- Bad, because it keeps digital-only cards (e.g. ~750 Alchemy cards) in the index that most requests then filter out at retrieval, a little wasted index space in exchange for not encoding format policy
