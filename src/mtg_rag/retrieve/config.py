"""Retrieval-filter configuration: values only.

No format table and no hardcoded format list. Ingest writes one `legal_<format>`
column per format Scryfall reports ([ADR 0001]), so the column a format maps to
is `f"{LEGALITY_COLUMN_PREFIX}{format_name}"` by construction, and the set of
formats is read off the frame's columns rather than restated here.
"""

from __future__ import annotations

#: Prefix `ingest.build_frame` gives each per-format legality column.
LEGALITY_COLUMN_PREFIX = "legal_"

#: Legality states that let a card into a deck. `restricted` is playable — it
#: caps you at one copy, a deckbuilding rule rather than a ban — and it appears
#: in several eternal formats, not just vintage, so one global set covers every
#: format. The excluded states are `banned` and `not_legal`.
PLAYABLE_LEGALITIES: frozenset[str] = frozenset({"legal", "restricted"})

#: Platform assumed when the user picks none. Paper is the common case, and the
#: one that makes the strictest, safest default (digital-only cards drop out).
DEFAULT_PLATFORM = "paper"
