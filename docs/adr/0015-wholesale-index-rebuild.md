---
status: "accepted"
date: 2026-07-20
---

# Rebuild the vector index wholesale; no incremental reconciliation

## Context and Problem Statement

[ADR 0010](0010-oracle-id-identity-key.md) left one question open and pointed it here: when the parquet is rebuilt on a refresh, how does the vector index catch up? Scryfall ships full daily snapshots, so a refresh rewrites the whole parquet ([ADR 0009](0009-parquet-card-corpus.md)); most of what changes between snapshots is prices and legalities, which never touch the index at all ([ADR 0010](0010-oracle-id-identity-key.md)). The open part is the text: when a card's oracle or flavor text actually changes, or a card is added or removed, how does `embed/` reconcile the index with the parquet?

## Considered Options

- Rebuild every requested channel from scratch whenever the index is out of date
- Reconcile incrementally: detect which cards' text changed and insert / update / delete only those vectors

## Decision Outcome

Chosen option: "Rebuild wholesale", because the incremental machinery costs more than the thing it saves and introduces a failure mode the rebuild cannot have.

`just embed` rebuilds every requested channel from the current parquet. A sidecar records what the index was built from — the model id, the dimension, and the corpus snapshot it was embedded from — and answers exactly one question: is this index current for this parquet and this model? If yes, it is a no-op; if no, it rebuilds. There is no diff and no per-card reconciliation.

Incremental reconciliation is what one reaches for, and it is the wrong trade here. Detecting *which* cards changed needs a per-channel content hash stored somewhere, and [ADR 0010](0010-oracle-id-identity-key.md) already declined to add hash columns to the parquet on the grounds that they are forward-compatibility scaffolding for machinery that does not exist — the guardrail this project holds. Acting on the diff needs three code paths — insert new cards, re-embed changed ones, delete removed ones — and each carries a way for the index to silently disagree with the parquet: a missed delete leaves an orphan vector, a missed update leaves stale text embedded. A wholesale rebuild has one path and cannot drift, because it never carries state forward.

The cost that justifies incremental work simply isn't there. A full rebuild of all 88,268 vectors ([ADR 0013](0013-structural-card-predicate.md)) is single-digit minutes on the target hardware ([ADR 0012](0012-embedding-model.md)). And the common refresh changes no text at all, so under the wholesale rule it is a no-op that the sidecar detects in a snapshot comparison — the incremental machinery would exist to convert "a few minutes, rarely" into "seconds, rarely", which is not a trade worth new state and a new class of bug. Recovery is total: delete `data/`, run `just ingest && just embed`, and everything is restored, matching the ingest contract.

### Consequences

- Good, because the index cannot drift from the parquet — there is no incremental state to get wrong, and no orphan or stale vector is reachable
- Good, because it adds no hash columns and no reconciliation code, holding the no-scaffolding guardrail ([ADR 0010](0010-oracle-id-identity-key.md) declined exactly this)
- Good, because "is the index current?" is one snapshot comparison in a sidecar, mirroring the ingest idempotency check rather than inventing a new mechanism
- Bad, because a one-card oracle errata re-embeds all 88,268 vectors, not the one that changed — wasteful in isolation, but cheap in absolute terms and paid rarely
- Bad, because there is a floor on refresh cost: even a no-text-change refresh that does trip the sidecar (e.g. a model change) pays the full rebuild, with no fast path for "almost nothing moved"
