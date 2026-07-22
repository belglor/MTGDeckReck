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

[ADR 0017] settles the fourth case, which ADR 0013 answered the other way: an
object missing `layout` or `set_type` is not a card here, because those two
fields are the entire basis for the judgement.
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

    An absent value is *not* a real card ([ADR 0017]): these two fields are what
    classifies the object, so without them there is no basis to call it a card.
    Ingestion refuses to project such a printing at all, which makes this
    unreachable from `just ingest` and leaves it guarding frames built by other
    routes — a notebook, a test — where the alternative is polars' null
    propagation dropping the row for an unrelated reason.
    """
    if layout is None or set_type is None:
        return False
    return layout not in EXCLUDED_LAYOUTS and set_type not in EXCLUDED_SET_TYPES


def is_real_card() -> pl.Expr:
    """A polars predicate selecting real, deckable cards.

    Composes into `frame.filter(is_real_card())`. Structural only: it reads
    `layout` and `set_type` and nothing else.

    `fill_null(True)` makes an absent value count as *excluded*, matching
    `is_real` ([ADR 0017]). It is spelled out rather than left to polars, whose
    three-valued logic would make `is_in` return null and `filter` drop the row —
    the same outcome, reached by accident instead of on purpose, and silently
    reversible by anyone who later rearranges the expression.
    """
    bad_layout = pl.col("layout").is_in(EXCLUDED_LAYOUTS).fill_null(True)
    bad_set_type = pl.col("set_type").is_in(EXCLUDED_SET_TYPES).fill_null(True)
    return ~(bad_layout | bad_set_type)


def real_cards(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only the real cards in `frame` ([ADR 0013])."""
    return frame.filter(is_real_card())
