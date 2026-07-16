# Architecture Decision Records

This directory records the significant architectural decisions for this project — why a choice was made, what alternatives were considered, and what trade-off was accepted.

- One decision per file, named `NNNN-kebab-case-title.md` with a zero-padded, sequential 4-digit number.
- Copy `template.md` to start a new one.
- `status` starts at `proposed`; move to `accepted` once settled. If a later decision replaces one, mark the old one `superseded by ADR-NNNN` rather than editing or deleting it — ADRs are an immutable log, not living documentation.
- Keep each ADR scoped to one decision; don't bundle unrelated choices together.

## Index

- [0001](0001-legality-color-as-filters-not-prompts.md) — Format legality and color identity are retrieval filters, not planner/LLM concerns
