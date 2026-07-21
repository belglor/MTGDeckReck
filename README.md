# MTGDeckReck

[![CI](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml/badge.svg)](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml)

Agent-driven RAG project for Magic: The Gathering deckbuilding recommendations.

## Development

`just` is the command surface — see `justfile`:

- `just setup` — sync dependencies, install pre-commit hooks
- `just check` — lint, typecheck, test; exactly what CI runs
- `just ingest` — build the card corpus (see below)
- `just embed` — build the vector index (see below)
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

## Vectors

`just embed` turns `data/cards.parquet` into `data/vectors/` — one Chroma
collection per embedding channel (oracle text with the card name folded in,
flavor text, and type line), keyed by `oracle_id`. Structural non-cards — tokens,
emblems, planes, schemes, vanguards — are excluded, leaving roughly 34k cards and
88k vectors. A card with no text in a channel simply has no entry there.

The model is an **optional dependency**, because it pulls torch and only the
machine building the index needs it:

```sh
uv sync --extra embed   # ~2.5 GB; not installed by `just setup`
just embed              # first run also downloads ~1.2 GB of model weights
```

Like `just ingest` it is manual and idempotent: a re-run compares
`data/vectors.meta.json` against the corpus snapshot, the model id, and the
vector dimension, and does nothing if the index is already current — without
loading the model. `just embed --force` rebuilds anyway. `just embed --channel
flavor` rebuilds a single channel while iterating; a partial run deliberately
leaves the sidecar alone, since it cannot establish that the whole index is
current. The index is rebuilt wholesale rather than reconciled card by card.

The model id and vector dimension live in `src/mtg_rag/embed/config.py`. Changing
either invalidates the index and starts a new retrieval baseline rather than
continuing the old one. Compute dtype is detected from the hardware.

URLs, file names, and separators the ingester depends on live in
`src/mtg_rag/ingest/config.py`. The one value read from the environment is
`SCRYFALL_USER_AGENT` — Scryfall asks API clients to identify themselves with
real contact info, which doesn't belong hardcoded in source. Copy
`.env.example` to `.env` (gitignored) to set it locally.

### Review

`@autoreview` is this repo's automated review process: **comment it on a pull
request** and a multi-agent Claude review runs against the diff, posting only
findings it scores as high-confidence. Restricted to repo collaborators.

It runs on demand rather than on every push, because each invocation costs real
quota. On demand is not the same as optional — it is the accepted way changes get
reviewed here; it simply isn't wired up as a merge gate.

You can also mention `@claude` on any issue or PR to ask a question or hand off a task.
