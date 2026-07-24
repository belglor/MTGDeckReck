# Architecture Decision Records

This directory records the significant architectural decisions for this project — why a choice was made, what alternatives were considered, and what trade-off was accepted.

## Conventions

- One decision per file, named `NNNN-kebab-case-title.md` with a zero-padded, sequential 4-digit number.
- Copy `template.md` to start a new one.
- `status` starts at `proposed`; move to `accepted` once settled. If a later decision replaces one, mark the old one `superseded by ADR-NNNN` rather than editing or deleting it — ADRs are an immutable log, not living documentation.
- Keep each ADR scoped to one decision; don't bundle unrelated choices together.

## Index

- [0001](0001-legality-color-as-filters-not-prompts.md) — Format legality and color identity are retrieval filters, not planner/LLM concerns
- [0002](0002-one-card-one-record.md) — One card is one record; no document chunking
- [0003](0003-sectioned-format-templates.md) — Deck-building guidance lives in hand-maintained, sectioned format templates
- [0004](0004-planner-typed-query-schema.md) — The planner emits a typed query schema; the model chooses the queries
- [0005](0005-curation-groups-by-role.md) — Curation returns cards grouped by role with theme-fit rationale
- [0006](0006-eval-measures-retrieval-recall.md) — Evaluation measures retrieval recall on the candidate pool, not final recommendations
- [0007](0007-multi-channel-embedding.md) — Embed card properties as separate semantic channels; structured facts stay filters
- [0008](0008-rrf-fusion-not-raw-scores.md) — Combine channel rankings with Reciprocal Rank Fusion, never by averaging raw scores
- [0009](0009-parquet-card-corpus.md) — The card corpus is stored as a single local parquet file
- [0010](0010-oracle-id-identity-key.md) — `oracle_id` is the identity key; the vector store holds vectors and nothing else
- [0011](0011-evaluation-scope-and-baseline-semantics.md) — Evaluation certifies retrieval recall in two modes; curation and flavor fidelity stay out of automated scope
- [0012](0012-embedding-model.md) — The embedding model is Qwen3-Embedding-0.6B, run locally
- [0013](0013-structural-card-predicate.md) — Exclude only structural non-cards from the index
- [0014](0014-absent-channel-no-vector.md) — An absent channel produces no vector, never a zero vector
- [0015](0015-wholesale-index-rebuild.md) — Rebuild the vector index wholesale; no incremental reconciliation
- [0016](0016-ingest-every-printing.md) — Ingest every printing and collapse it with an explicit representative-printing rule
- [0017](0017-structural-completeness-is-required.md) — A printing missing `layout` or `set_type` is refused at ingestion
- [0018](0018-card-level-platform-availability.md) — Platform availability is the union across a card's printings
- [0019](0019-flavor-channel-embeds-one-text.md) — The flavor channel embeds one text per card: the most recent printing that has any
- [0020](0020-eval-case-is-a-corpus-predicate.md) — An eval case is a corpus predicate measured as lift, not a list of expected cards
