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

# Download the Scryfall oracle-cards snapshot into data/cards.parquet
ingest *args:
    uv run python -m mtg_rag.ingest {{args}}

# Embed the corpus into data/vectors/ (one Chroma collection per channel).
# Needs the optional model half: uv sync --extra embed
embed *args:
    uv run python -m mtg_rag.embed {{args}}

notebook:
    uv run jupyter lab
