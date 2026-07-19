---
status: "accepted"
date: 2026-07-19
---

# Combine channel rankings with Reciprocal Rank Fusion, never by averaging raw scores

## Context and Problem Statement

Each embedding channel ([ADR 0007](0007-multi-channel-embedding.md)) returns its own ranked list of cards for a query, and the planner issues several queries per request ([ADR 0004](0004-planner-typed-query-schema.md)). These have to become one candidate pool before curation sees it. How are independent rankings combined?

## Considered Options

- Reciprocal Rank Fusion over each ranking's ordinal positions
- Average the raw similarity scores across channels
- Take each card's maximum score across channels
- Skip fusion: search one channel and use the others only as tie-breakers

## Decision Outcome

Chosen option: "Reciprocal Rank Fusion", scoring each card as the sum over channels of `1 / (k + rank)` with `k = 60`.

Averaging raw scores is the option worth arguing against explicitly, because it is what one reaches for first and it is wrong here. Cosine similarities from different channels are not commensurable. Each channel embeds a different register of text over a different corpus — the flavor channel indexes a fraction of the cards the oracle channel does — so the channels have different score distributions and no shared calibration. A 0.8 from the flavor channel and a 0.8 from the oracle channel are not the same claim, and averaging them silently asserts that they are. Maximum-score fusion has the same defect and additionally throws away agreement between channels, which is the most useful signal available.

RRF uses only ordinal position, which is comparable across channels by construction: rank 1 means "this channel's best match" regardless of how the channel scores. It needs no calibration, no normalization step, and no per-channel score statistics.

**Channels are weighted uniformly for now.** The known weakness of unweighted RRF is dilution: every channel contributes its top-k whether or not it had anything relevant, so on a precise query like "urzatron lands" the flavor channel contributes noise at the same weight as the oracle channel that actually holds the answer. This is accepted rather than pre-emptively fixed, because any weights chosen today would be guesses. The golden set in [ADR 0006](0006-eval-measures-retrieval-recall.md) is the instrument for deciding whether weighting is needed and what the weights should be — measure the dilution first, then correct it.

### Consequences

- Good, because fusion is scale-free — channels can use different embedding models or be re-embedded independently without recalibrating anything
- Good, because a channel that is useless for a given query degrades the pool gently instead of poisoning it, since its contribution is bounded by rank
- Good, because cards ranked well by several channels rise above cards ranked well by one, which is usually the right instinct for theme fit
- Good, because fusion is deterministic and cheap — no model call, and the same inputs always produce the same pool
- Bad, because rank discards magnitude: a channel that is confidently correct and a channel that is weakly guessing contribute identically at the same rank
- Bad, because dilution on precise queries grows with channel count, and is a real cost being knowingly deferred rather than solved
- Bad, because `k = 60` is inherited convention rather than a tuned value for this corpus, and nothing yet justifies it beyond it being the standard default
