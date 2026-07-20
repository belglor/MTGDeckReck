---
status: "accepted"
date: 2026-07-20
---

# An absent channel produces no vector, never a zero vector

## Context and Problem Statement

The three channels ([ADR 0007](0007-multi-channel-embedding.md)) do not all cover the same cards. Of the 34,184 real cards ([ADR 0013](0013-structural-card-predicate.md)), oracle text covers 99.0%, type line 100%, but flavor text only 59.2% — **13,936 real cards have no flavor text at all**, and 348 vanilla cards have no oracle text. So for a large fraction of the corpus, at least one channel has nothing to embed. What does a channel store for a card that has no text in it?

## Considered Options

- Store no vector: a card with no text in a channel simply has no entry in that channel
- Store a zero vector as a placeholder so every channel holds every card
- Embed a fallback string (e.g. the card name, or the type line) so the channel is never empty

## Decision Outcome

Chosen option: "Store no vector", because absence is the honest representation and the two alternatives both inject signal that isn't there.

A card with no text in a channel gets no id and no vector in that channel's collection. It is not absent from the corpus — the parquet still holds it, and it still has vectors in the channels it does populate.

The zero-vector placeholder is the option worth arguing against, because it looks harmless and is not. A zero vector is not a neutral point; under cosine similarity it degenerates, and every zero vector is equidistant from every query, so the 13,936 flavorless cards would form one mutual-neighbour blob that surfaces together in the flavor channel regardless of the query. That is noise presented as matches. Absence composes correctly instead: fusion is over ordinal rank ([ADR 0008](0008-rrf-fusion-not-raw-scores.md)), and a card that is not in a channel's ranking contributes nothing from that channel — which is exactly true, rather than a false zero.

The fallback string was rejected for the same reason [ADR 0007](0007-multi-channel-embedding.md) folds the name into oracle text but nowhere else: each channel embeds one register of language, and injecting a name or a type line into the flavor channel would put a vector there in a register unlike every other vector in the collection, diluting the very signal the channel exists to isolate. A vanilla creature with no oracle text is reachable through its type and flavor channels; it does not need a manufactured oracle vector to be found.

This raises a fair worry: does a card that populates fewer channels get penalised in the fused ranking? It does not, and the distinction is the point. A missing channel contributes **0** to the fused score ([ADR 0008](0008-rrf-fusion-not-raw-scores.md)), not a negative — the card is not marked "bad" in that channel, it simply does not compete there, while competing fully in the channels it does populate. A vanilla `Creature — Bear` ranks at the top of the *type* channel for a bears query and reaches the candidate pool through that rank, which is what the recall metric actually measures ([ADR 0006](0006-eval-measures-retrieval-recall.md)) — top-of-pool is not the bar, reaching the pool is. Where thematic recall for a sparse card needs help, the lever is the planner emitting a query the card is relevant to ([ADR 0004](0004-planner-typed-query-schema.md)), whose results enter the pool by union — not a per-card fusion adjustment. Normalising fusion by how many channels a card populates is precisely the score-averaging [ADR 0008](0008-rrf-fusion-not-raw-scores.md) rejects: it would let a card strong in one channel outrank a card strong in three, promoting sparse commodity cards over rich thematic ones, backwards for a theme recommender.

### Consequences

- Good, because the flavor channel indexes only cards that have flavor text, so its neighbours are real matches rather than a blob of placeholders
- Good, because absence contributes nothing to fusion instead of a false zero, which is what rank-based RRF ([ADR 0008](0008-rrf-fusion-not-raw-scores.md)) assumes
- Good, because collection sizes differ by channel and that is fine — the store keys everything by `oracle_id` ([ADR 0010](0010-oracle-id-identity-key.md)) and never assumes a card is present in every channel
- Bad, because a card populating fewer channels has fewer chances to accumulate fused score, so a rich card outranks a sparse one on any single query — usually the right instinct for a theme deck, but it means sparse cards depend on the planner ([ADR 0004](0004-planner-typed-query-schema.md)) routing a query to the channel where they do have signal
- Bad, because code reading the store must treat "no vector for this id in this channel" as normal, not an error — a latent bug for anyone who assumes every real card appears in every collection
