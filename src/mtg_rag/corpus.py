"""The structural "is this a real card" predicate, shared by `ingest/`,
`embed/` and `retrieve/`.

It comes in two shapes over one definition: `is_real_card` as a polars
expression for filtering a frame, and `is_real` for a single card's values,
which ingestion needs while choosing between a card's printings and before any
frame exists. Both read the same exclusion lists, and a test pins them to the
same answers.

Ingestion keeps every row Scryfall ships ([ADR 0009]): tokens, emblems,
art-series prints, planes, schemes, vanguards, and a handful of `layout: normal`
non-cards (Celebration / Collectors' Edition memorabilia) ride along with the
real cards. This predicate removes exactly those structural non-cards and
nothing else — no legality, no `games`, no color identity. Those are
deterministic retrieval-time filters ([ADR 0001]) that the parquet owns; folding
any of them in here would make the index encode policy and force a re-embed when
policy changes ([ADR 0010]).

See [ADR 0013] for the exclusion lists, the measured row counts, and the three
traps this shape exists to avoid: planes / schemes / vanguards are excluded by
`layout`, never `set_type` (`set_type == "planechase"` holds real cards); Un-set
cards stay (silver-border legality is legality, not structure); Alchemy and other
digital-only cards stay (digital-ness is what the `games` filter is for).
"""

from __future__ import annotations

import polars as pl

from mtg_rag.corpus_config import EXCLUDED_LAYOUTS, EXCLUDED_SET_TYPES


def is_real(layout: str | None, set_type: str | None) -> bool:
    """`is_real_card` for one card's values instead of a frame's columns.

    Ingestion needs this shape before a frame exists: `merge` picks which
    printing represents a card, and a printing that is not itself a real card
    must not be allowed to win. Tundra's most recent printing is 30th
    Anniversary Edition — `set_type: memorabilia` — so ranking purely by date
    would stamp a memorabilia set type onto a real card and drop it out of the
    index entirely, along with 300-odd others.

    Null is treated as not-excluded, matching `is_real_card`'s `fill_null(False)`
    for the same reason: a card is dropped only when a value actively matches an
    exclusion list, never merely because a field is absent.
    """
    bad_layout = layout is not None and layout in EXCLUDED_LAYOUTS
    bad_set_type = set_type is not None and set_type in EXCLUDED_SET_TYPES
    return not (bad_layout or bad_set_type)


def is_real_card() -> pl.Expr:
    """A polars predicate selecting real, deckable cards.

    Composes into `frame.filter(is_real_card())`. Structural only: it reads
    `layout` and `set_type` and nothing else. A null value is treated as
    not-excluded (`fill_null(False)`), so a card is dropped only when a value
    actively matches an exclusion list — never merely because a field is absent,
    which unguarded null propagation through `filter` would otherwise do.
    """
    bad_layout = pl.col("layout").is_in(EXCLUDED_LAYOUTS).fill_null(False)
    bad_set_type = pl.col("set_type").is_in(EXCLUDED_SET_TYPES).fill_null(False)
    return ~(bad_layout | bad_set_type)


def real_cards(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only the real cards in `frame` ([ADR 0013])."""
    return frame.filter(is_real_card())
