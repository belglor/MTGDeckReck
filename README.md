# MTGDeckReck

[![CI](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml/badge.svg)](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml)

Recommends Magic: The Gathering cards from a plain-English description of the deck
you want to build. It is aimed at players who build for flavor and theme rather
than for tournament results, so it answers requests like:

- *"A commander deck with a spooky graveyard theme."*
- *"Here's my current deck — do I have too much ramp? Should I add removal?"*

New to the vocabulary? See the [glossary](docs/glossary.md).

## How it works

A request passes through five stages. The first two build local data files; the
rest run per request.

| Stage | What it does | Built? |
|---|---|---|
| **Ingest** | Downloads Scryfall's card data into a local table | yes |
| **Embed** | Turns each card's text into vectors for semantic search | yes |
| **Retrieve** | Filters to legal, in-color cards, then searches for candidates | yes |
| **Plan** | Asks an LLM what to search for, given the deck request | no |
| **Curate** | Asks an LLM to group the candidates by role and explain the picks | no |

What each stage does and why: [`docs/spec.md`](docs/spec.md). The reasoning behind
individual decisions: [`docs/adr/`](docs/adr/).

## Getting started

```sh
just setup     # install dependencies and pre-commit hooks
just ingest    # build the card table (a few minutes)
just embed     # build the search index (downloads a model; see below)
just retrieve "spooky graveyard recursion" --colors B   # search it
```

`just` is the command surface — run `just` or read the `justfile` for the full list.

Both build steps are manual; nothing refreshes on a schedule. Both are also
idempotent: each records what it built in a sidecar file and does nothing if that
still matches. Pass `--force` to rebuild anyway.

Everything lands in `data/`, which is gitignored. Delete it and re-run to restore.

| File | Holds |
|---|---|
| `data/cards.parquet` | one row per card, ~38k rows, ~4 MB |
| `data/cards.meta.json` | which Scryfall snapshot the table came from |
| `data/vectors/` | one Chroma collection per channel, ~88k vectors |
| `data/vectors.meta.json` | which table and model the vectors came from |

## Card data

`just ingest` downloads Scryfall's `default_cards` bulk snapshot and writes
`data/cards.parquet`.

That snapshot holds one object per *printing* — 116k of them for 38k cards —
because some facts differ between printings and cannot be read off just one.
Ingestion keeps the English printings and collapses them to one row per card: the
most recent real printing supplies the card's set, rarity, release date and
prices, so those always describe one physical object. See
[ADR 0016](docs/adr/0016-ingest-every-printing.md).

Two fields ignore that rule, because for them no single printing has the right
answer:

- **Flavor text** comes from the most recent printing that actually has any.
  Otherwise 1,800 cards whose latest reprint carries none would lose it.
- **Platforms** — paper, MTGO, Arena — are the union across every printing.
  Otherwise 1,042 commander-legal paper cards would look MTGO-only, because
  Scryfall happened to pick an MTGO-only reprint to represent them. See
  [ADR 0018](docs/adr/0018-card-level-platform-availability.md).

Scryfall asks API clients to identify themselves with real contact details, which
don't belong in committed source. Copy `.env.example` to `.env` (gitignored) and
set `SCRYFALL_USER_AGENT`. URLs, file names and separators live in
[`src/mtg_rag/ingest/config.py`](src/mtg_rag/ingest/config.py).

## Vectors

`just embed` turns `data/cards.parquet` into `data/vectors/` — one Chroma
collection per channel, keyed by `oracle_id`. The three channels are oracle text
(with the card name folded in), flavor text, and type line. Structural non-cards
such as tokens and emblems are excluded, leaving roughly 34k cards and 88k
vectors. A card with no text in a channel simply has no entry there.

`just setup` installs everything, the model included — retrieval encodes each
query with the same model, so it is a core dependency. The first `just embed`
also downloads ~1.2 GB of model weights.

`just embed --channel flavor` rebuilds one channel while iterating; a partial run
leaves the sidecar alone, since it cannot establish that the whole index is
current. The index is rebuilt wholesale rather than reconciled card by card.

The model id and vector dimension live in
[`src/mtg_rag/embed/config.py`](src/mtg_rag/embed/config.py). Changing either
invalidates the index and starts a new retrieval baseline rather than continuing
the old one.

## Retrieval

`just retrieve` searches the index and prints a candidate pool. It exists so the
retrieval path is usable before the planner that will eventually drive it:

```sh
just retrieve "spooky graveyard recursion" "self-mill enablers" \
    --format commander --colors B --explain
```

Each quoted phrase is one query. Every query is run against all three channels,
and the rankings are fused into one pool by Reciprocal Rank Fusion — combining
positions, never raw similarity scores, which are not comparable between
channels ([ADR 0008](docs/adr/0008-rrf-fusion-not-raw-scores.md)). `--explain`
shows which query, channel and rank found each card.

Before any search runs, the hard constraints you pick — format legality, color
identity, platform — filter the corpus down to the cards you are allowed to see,
and the search is restricted to those
([ADR 0001](docs/adr/0001-legality-color-as-filters-not-prompts.md)). An illegal
or off-color card cannot come back, rather than being filtered out afterwards.

`--colors` omitted means no color constraint; `--colors ""` means colorless only.
Other flags: `--platform`, `--channel` (repeatable, to see one channel alone),
`-k` for pool size, `--data-dir`.

An empty pool is a valid answer — a colorless deck asking for black removal is
honestly unsatisfiable.

## Development

`just check` runs lint, typecheck and tests — exactly what CI runs.

`main` is protected: changes land through a pull request with green CI, and a
human makes the call to merge.

Two notebooks explore the data rather than serving it:
`notebooks/01-scryfall-exploration.ipynb` profiles the corpus, and
`notebooks/02-embedding-exploration.ipynb` inspects the vectors. Commit notebooks
without outputs; the `nbstripout` pre-commit hook enforces this.

### Review

Comment `@autoreview` on a pull request and a multi-agent Claude review runs
against the diff, posting only high-confidence findings. It is restricted to repo
collaborators and runs on demand rather than on every push, because each run costs
real quota — on demand is not the same as optional, it simply isn't a merge gate.

You can also mention `@claude` on any issue or PR to ask a question or hand off a
task.
