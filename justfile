# Recipe arguments reach the command as "$@" rather than being re-split on
# whitespace. `just retrieve "graveyard recursion"` must arrive as one query,
# not three — {{args}} would flatten the quoting away.
set positional-arguments

setup:
    uv sync
    uv run pre-commit install

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run pyright

test:
    uv run pytest

check: lint typecheck test

# Download the Scryfall bulk snapshot into data/cards.parquet
ingest *args:
    uv run python -m mtg_rag.ingest "$@"

# Embed the corpus into data/vectors/ (one Chroma collection per channel).
# The first run downloads ~1.2 GB of model weights.
embed *args:
    uv run python -m mtg_rag.embed "$@"

# Search the index and print a fused candidate pool.
# e.g. just retrieve "graveyard recursion" --colors B --explain
retrieve *args:
    uv run python -m mtg_rag.retrieve "$@"

notebook:
    uv run jupyter lab
