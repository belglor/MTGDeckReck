"""Ingestion configuration.

Values that named a URL, a file name, or a join separator used to be defined
wherever they were first used — `scryfall.py`, `normalize.py`, `__main__.py` —
which meant a change to any of them was a grep across the package. They live
here instead, so it's a one-line diff.

`USER_AGENT` is the one value read from the environment rather than hardcoded:
Scryfall asks API clients for real contact info in it, and a maintainer's
contact details don't belong committed to source, even as a public GitHub URL.
Set `SCRYFALL_USER_AGENT` — in the process environment or in a local `.env`
(see `.env.example`) — to send one; the default carries none.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path: Path) -> None:
    """Populate `os.environ` from a `KEY=VALUE` file, without overriding
    variables the environment already set.

    Stdlib-only: pulling in `python-dotenv` for ~15 lines of parsing isn't
    worth a new runtime dependency.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


load_dotenv(_REPO_ROOT / ".env")

#: Scryfall asks API clients to identify themselves; requests without a
#: User-Agent are rejected outright.
USER_AGENT = os.environ.get("SCRYFALL_USER_AGENT", "MTGDeckReck/0.1")

#: Scryfall's bulk-data index, listing where to download each bulk export.
BULK_INDEX_URL = "https://api.scryfall.com/bulk-data"

#: Which bulk-data export to ingest by default.
DEFAULT_BULK_TYPE = "oracle_cards"

#: Separator for text joined across a card's faces. Newlines keep the halves
#: visually distinct in oracle and flavor text, matching how Scryfall renders
#: split cards.
FACE_SEPARATOR = "\n//\n"

#: Mana costs are joined inline instead, since a newline inside a cost string
#: would be nonsense.
COST_SEPARATOR = " // "

#: On-disk names for the corpus and its provenance sidecar, under whatever
#: --data-dir the CLI is given.
CORPUS_NAME = "cards.parquet"
SIDECAR_NAME = "cards.meta.json"
