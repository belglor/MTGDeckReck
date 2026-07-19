---
status: "accepted"
date: 2026-07-19
---

# `oracle_id` is the identity key; the vector store holds vectors and nothing else

## Context and Problem Statement

A card's data could end up in two places: the corpus parquet ([ADR 0009](0009-parquet-card-corpus.md)) and the vector store holding the three embedding channels ([ADR 0007](0007-multi-channel-embedding.md)). Two copies of anything invites drift, and cards do change — banned-list announcements move legality, prices move daily, oracle text is occasionally errata'd.

Two questions follow. What key ties a vector unambiguously to its card? And when a card changes, what has to be rewritten before that change reaches the user?

The second question is sharpened by [ADR 0001](0001-legality-color-as-filters-not-prompts.md), which requires legality and color identity to be applied as deterministic filters *before* any LLM sees a candidate. Something has to evaluate those filters, and where that happens decides how much state the vector store has to carry.

## Considered Options

- Key on Scryfall's `id`, the printing identifier
- Key on card name
- Key on `oracle_id`, and mirror the full record into the vector store so retrieval is a single call
- Key on `oracle_id`, and mirror the filter fields into the vector store as searchable metadata
- Key on `oracle_id`, and store no metadata at all — filter the parquet first, then constrain the search to the surviving ids

## Decision Outcome

Chosen option: "Key on `oracle_id`, and store no metadata at all".

**On the key.** Scryfall's `id` identifies a *printing*, and the `oracle_cards` bulk file explicitly re-picks "the most up-to-date recognizable version" of each card on every build. So `id` — along with `set`, `released_at`, `flavor_text`, and `prices` — can change between two refreshes for a card whose rules text never moved. Keying vectors on `id` would silently orphan them on a routine refresh, in a way that looks like a missing card rather than a broken join. Names are worse: they collide across faces and are not stable identifiers. `oracle_id` is stable across printings and is one-per-card, exactly the granularity [ADR 0002](0002-one-card-one-record.md) chose as the retrieval unit.

**On what the vector store holds.** Nothing but vectors and their `oracle_id`. Retrieval runs in two steps:

1. Filter the corpus parquet — the single source of truth — with the deterministic [ADR 0001](0001-legality-color-as-filters-not-prompts.md) constraints, producing the set of `oracle_id`s the user is allowed to see.
2. Hand that set to the vector store as an explicit id allowlist, so the search is constrained before ranking.

The card text handed to the curation call is then read from the parquet by `oracle_id`. The vector store never holds a copy of anything the LLM sees.

The rejected alternative was mirroring the filter fields into the vector store as searchable metadata. It works, but it is both slower and more state. Measured against Chroma 1.5.9 with 38,312 vectors at dim 384, `n_results=50`:

| Approach | per search | per request (3 channels × 8 planner queries) |
|---|---|---|
| Mirrored metadata filter (`where=`) | 48.7 ms | 1,170 ms |
| Id allowlist of 30,000 (`ids=`) | 8.3 ms | 199 ms |

The allowlist is faster despite carrying every surviving id, and it is exact — no result fell outside the allowlist. Two caveats on those numbers: they were taken on synthetic random vectors with an in-memory client, so real embeddings and a persistent store will differ in absolute terms; and the reason the metadata path costs ~6× more was measured, not explained.

A related question was settled at the same time. Chroma **pre-filters**: with 1% of rows flagged and deliberately placed far from the query vector while the other 99% sat close, an unfiltered top-10 returned none of the flagged rows, yet the filtered query returned a full 10, all flagged. Post-filtering could not produce that. This matters because [ADR 0001](0001-legality-color-as-filters-not-prompts.md) would otherwise hold only at the cost of off-color cards eating the top-k, forcing every query to over-fetch by an unknown factor.

**On change propagation.** [ADR 0007](0007-multi-channel-embedding.md) already forbids embedding structured facts, so the fields most likely to change are never inside a vector. Combined with a vector store that holds no metadata, refresh reduces to two cases:

| What changed | How it is detected | What it costs |
|---|---|---|
| Legality, price, rarity, set | filter fields differ; channel text unchanged | **nothing** — rewriting the parquet is the whole job |
| Oracle or flavor text errata'd | that channel's text differs | re-embed that one channel |
| Card added or removed | `oracle_id` present on only one side | insert or delete its vectors |

A typical daily Scryfall snapshot moves prices on thousands of cards and oracle text on approximately none, so the common refresh touches the vector store not at all. This directly addresses the stale-data risk [ADR 0001](0001-legality-color-as-filters-not-prompts.md) accepted as its only downside: there is no second copy of legality that could lag behind the first.

Comparing per-channel content hashes is the intended detection mechanism for the text case. Hashes are derivable from the parquet whenever they are needed, so ingestion does **not** store them today — adding columns nothing reads yet would be exactly the forward-compatibility scaffolding this project avoids.

Where the reconciliation logic lives is still open, and should be decided with `embed/`. Note that the allowlist design makes the Chroma-versus-FAISS choice less consequential than it would otherwise be: FAISS has no metadata index at all, which would have ruled out the mirrored-metadata approach outright, but an id allowlist reduces to a mask over a subset — and exact cosine over ~30k filtered vectors at dim 768 measured 19.8 ms, so brute force remains on the table at this corpus size.

### Consequences

- Good, because a legality or price change reaches the LLM the moment the parquet is rewritten, with no embedding work, no re-indexing, and no metadata update
- Good, because there is exactly one copy of every user-visible and filterable value, so "is this current?" has one answer rather than two that must agree
- Good, because the filters are ordinary expressions over a dataframe rather than a query language the vector store happens to support, so adding one is a code change in one place
- Good, because `oracle_id` survives Scryfall re-picking a card's printing, which is routine rather than an edge case
- Bad, because retrieval needs a second lookup — vector search returns ids, and the record still has to be fetched from the corpus before an LLM call
- Bad, because every query serializes an allowlist that may run to tens of thousands of ids; this is cheap in-process and would not be if the vector store ever moved behind a network boundary, which this design therefore assumes it will not
- Bad, because the corpus must be loaded and filterable at request time, so the parquet stops being purely an ingestion artifact and becomes a live dependency of the request path — a constraint [ADR 0009](0009-parquet-card-corpus.md) does not currently anticipate
- Bad, because between a text errata and the next re-embed, retrieval scores against an old vector while display shows new text — tolerable, but real, and only detectable if something actually compares the hashes
