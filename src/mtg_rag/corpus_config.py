"""Exclusion lists for the structural "is this a real card" predicate
([ADR 0013]).

Kept separate from `corpus.py` so the predicate's logic and its data can
change independently — a new memorabilia set type is a one-line diff here,
not a change to the function that reads it.
"""

from __future__ import annotations

#: Oversized or non-deck layouts — objects a player never puts in a deck.
EXCLUDED_LAYOUTS: frozenset[str] = frozenset(
    {
        "art_series",
        "token",
        "double_faced_token",
        "emblem",
        "vanguard",
        "scheme",
        "planar",
        "augment",
        "host",
    }
)

#: Set types whose `layout: normal` contents are non-cards nothing else catches
#: — Celebration / Collectors' Edition memorabilia, token sets, minigame inserts.
#: Deliberately small: layout exclusion alone already removes almost everything,
#: and this list adds only the handful of normal-layout stragglers ([ADR 0013]).
EXCLUDED_SET_TYPES: frozenset[str] = frozenset(
    {
        "memorabilia",
        "token",
        "minigame",
    }
)
