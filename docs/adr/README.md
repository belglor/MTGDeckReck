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
