"""Which rows are real cards, and which are tokens, emblems and other props.

Ingestion keeps everything Scryfall ships ([ADR 0009]), so the corpus holds both.
This predicate separates them on `layout` and `set_type` alone. Legality, colour
and platform are retrieval-time filters ([ADR 0001]) and deliberately not part of
the judgement — folding one in would make the index encode policy.

Two shapes over one definition: `is_real_card` filters a frame, `is_real` answers
for a single card's values. A test pins them to the same answers.

[ADR 0013] has the exclusion lists and the cases that make them subtle.
[ADR 0017] adds that a missing `layout` or `set_type` is not a card.
"""

from __future__ import annotations

import polars as pl

from mtg_rag.corpus_config import EXCLUDED_LAYOUTS, EXCLUDED_SET_TYPES


def is_real(layout: str | None, set_type: str | None) -> bool:
    """`is_real_card` for one card's values instead of a frame's columns.

    `ingest.merge` needs the check before a frame exists, to stop a non-card
    printing from representing a real card ([ADR 0016]).

    An absent value is not a real card ([ADR 0017]): these two fields are the
    whole basis for the judgement, so without them there is nothing to judge on.
    """
    if layout is None or set_type is None:
        return False
    return layout not in EXCLUDED_LAYOUTS and set_type not in EXCLUDED_SET_TYPES


def is_real_card() -> pl.Expr:
    """A polars predicate selecting real, deckable cards.

    Composes into `frame.filter(is_real_card())`.

    `fill_null(True)` excludes rows with an absent value, matching `is_real`
    ([ADR 0017]). Polars' null propagation would drop them anyway, but by
    accident rather than on purpose — and anyone rearranging this expression
    would silently lose the behaviour.
    """
    bad_layout = pl.col("layout").is_in(EXCLUDED_LAYOUTS).fill_null(True)
    bad_set_type = pl.col("set_type").is_in(EXCLUDED_SET_TYPES).fill_null(True)
    return ~(bad_layout | bad_set_type)


def real_cards(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only the real cards in `frame` ([ADR 0013])."""
    return frame.filter(is_real_card())
