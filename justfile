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
