# MTGDeckReck

[![CI](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml/badge.svg)](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml)

Agent-driven RAG project for Magic: The Gathering deckbuilding recommendations.

## Development

`just` is the command surface — see `justfile`:

- `just setup` — sync dependencies, install pre-commit hooks
- `just check` — lint, typecheck, test; exactly what CI runs
- `just ingest` — build the card corpus (see below)
- `just notebook` — open JupyterLab

`main` is protected: changes land through a pull request with green CI, and a human
makes the call to merge.

## Card data

`just ingest` downloads Scryfall's `oracle_cards` bulk snapshot and normalizes it
to `data/cards.parquet` (~38k cards, one row per card, ~4 MB). It is a manual
step — there is no scheduled refresh — and it is idempotent: a re-run checks the
upstream snapshot timestamp against `data/cards.meta.json` and does nothing if
the corpus is already current. Use `just ingest --force` to rebuild anyway.

`data/` is gitignored. Deleting it and re-running restores everything.

`notebooks/01-scryfall-exploration.ipynb` profiles the result — channel coverage,
corpus composition, and the multi-face joins. Commit notebooks without outputs;
the `nbstripout` pre-commit hook enforces this.

### Review

`@autoreview` is this repo's automated review process: **comment it on a pull
request** and a multi-agent Claude review runs against the diff, posting only
findings it scores as high-confidence. Restricted to repo collaborators.

It runs on demand rather than on every push, because each invocation costs real
quota. On demand is not the same as optional — it is the accepted way changes get
reviewed here; it simply isn't wired up as a merge gate.

You can also mention `@claude` on any issue or PR to ask a question or hand off a task.
