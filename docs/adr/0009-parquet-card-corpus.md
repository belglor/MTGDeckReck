---
status: "accepted"
date: 2026-07-19
---

# The card corpus is stored as a single local parquet file

## Context and Problem Statement

Ingestion pulls Scryfall's `oracle_cards` bulk file — roughly 180 MB of JSON, one object per Oracle ID — and has to put the normalized result somewhere on disk. Later stages read it: `embed/` to produce vectors, `evals/` to resolve golden-set card names, a notebook to explore it. What format holds the corpus?

## Considered Options

- A single parquet file
- A SQLite database file
- The raw Scryfall JSON, kept on disk as downloaded

## Decision Outcome

Chosen option: "A single parquet file", because the two things SQLite would buy us — partial updates and indexed `WHERE` clauses — turn out not to apply here.

Scryfall publishes a **full daily snapshot**, not a changelog. There is no delta to apply, so an upsert path would only be a more fragile way of expressing "replace the corpus": it can half-succeed and leave rows from two different snapshots side by side, where a whole-file rewrite either happens or doesn't. "Delete `data/` and re-run restores everything" is a property worth having, and immutable snapshot files have it for free.

Indexed lookups are the closer call, because this file *is* on the request path. Per [ADR 0010](0010-oracle-id-identity-key.md) the [ADR 0001](0001-legality-color-as-filters-not-prompts.md) filters are evaluated against this corpus to produce an id allowlist, and the card text shown to the LLM is read back from it. So "the query never touches this file" would be false.

It still favours parquet, for a different reason than not being queried. The request-path query is *"give me every `oracle_id` matching this predicate"* — tens of thousands of rows out of ~38k ([ADR 0002](0002-one-card-one-record.md)) — followed by a keyed lookup of a few dozen records. That is a columnar scan over a table small enough to hold in memory, which is precisely what parquet loaded into a dataframe is good at, and precisely the shape where SQLite's indexes earn least: an index accelerates finding *few* rows among many, while this predicate keeps most of them. Loading the file whole costs milliseconds and every subsequent filter is an in-memory expression with no round-trip.

Raw JSON was rejected on cost, not principle: every consumer would re-parse 180 MB and re-derive the same projection, including the multi-face text joining that is genuinely fiddly.

Note that SQLite was not rejected for being heavyweight — it is a stdlib module and a single file, with no server, no install, and no ops. It simply solves problems this corpus does not have.

### Consequences

- Good, because refresh is atomic in the only sense that matters: the file is either the old snapshot or the new one, never a mixture
- Good, because columnar storage means `embed/` can read just the three text channels without paying for prices and legalities, and the notebook can profile a single column cheaply
- Good, because it stays a dependency-free artifact — any tool that reads parquet can inspect the corpus without going through our code
- Bad, because every refresh rewrites the whole file even when one card's price moved
- Bad, because there is no query engine and no indexes; anything that wants a subset loads the file and filters in memory
- Bad, because the format assumes a single writer and no concurrent readers mid-write, which is fine for a manually-run script and would not be for a service
- Bad, because the whole corpus must be resident to serve a request ([ADR 0010](0010-oracle-id-identity-key.md)), so memory cost is paid up front and in full rather than per query

This decision should be revisited — and superseded — if the corpus outgrows comfortable memory, or if retrieval starts needing predicates selective enough that scanning every row to answer them becomes the bottleneck.
