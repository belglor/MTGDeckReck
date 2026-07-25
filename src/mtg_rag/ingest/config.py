"""Ingestion configuration — values only.

URLs, file names, and join separators live here rather than wherever they were
first used, so changing one is a one-line diff, not a grep across the package.

`USER_AGENT` is the exception read from the environment: Scryfall wants real
contact info in it, which doesn't belong in source. Set `SCRYFALL_USER_AGENT`
(process env or a local `.env`, see `.env.example`); the default carries none.
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

#: Which bulk-data export to ingest. `default_cards` carries **every printing**
#: — 116,138 objects against `oracle_cards`' 38,312 — because the facts that
#: differ between printings can only be answered by seeing all of them. What one
#: row of the corpus then contains is decided by `ingest.merge`, explicitly,
#: rather than inherited from whichever printing `oracle_cards` happened to pick.
BULK_TYPE = "default_cards"

#: The language a card must be printed in to enter the corpus. `default_cards`
#: falls back to a foreign printing for cards that have no English one at all,
#: and those would otherwise reach the embedding channels.
CORPUS_LANGUAGE = "en"

#: Where a card can actually be played, and the vocabulary the platform filter
#: offers. Scryfall also reports `astral` (10 cards, from the 1997 Shandalar PC
#: game) and `sega` (8 Dreamcast promos); both are deliberately absent, being
#: curiosities rather than places anyone plays. A card printed only there ends up
#: with no platforms and is invisible to the filter, which is intended
#: ([ADR 0018]).
PLATFORMS: tuple[str, ...] = ("paper", "mtgo", "arena")

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
