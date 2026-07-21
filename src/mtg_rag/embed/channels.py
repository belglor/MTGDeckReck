"""Per-channel text composition — pure, with no I/O and no model.

One row per card per channel, keyed by `oracle_id` ([ADR 0010]). A card with no
text in a channel produces **no row**, never a blank or placeholder one: a zero
vector is not a neutral point under cosine similarity, and a channel padded with
placeholders would surface them as one mutual-neighbour blob regardless of the
query. Absence composes correctly instead — a card missing from a channel's
ranking contributes nothing from that channel, which is exactly true rather than
a false zero ([ADR 0014]).

What counts as a card is the corpus predicate's decision alone, so
`channel_frame` applies it here rather than trusting every caller to remember
([ADR 0013]).
"""

from __future__ import annotations

import polars as pl

from mtg_rag.corpus import is_real_card
from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.embed.config import (
    CHANNEL_SOURCE_SEPARATOR,
    CHANNEL_SOURCES,
    TEXT_COLUMN,
    Channel,
)


def channel_expr(channel: Channel) -> pl.Expr:
    """The text expression for one channel, before absence filtering.

    Which columns a channel reads is configuration (`CHANNEL_SOURCES`); this
    only assembles them. `concat_str` propagates nulls by design, so a card
    missing any column its channel names composes to null here and is dropped by
    `channel_frame` rather than embedded as a partial string — which is what
    stops a vanilla card being indexed as a bare name ([ADR 0014]).
    """
    try:
        sources = CHANNEL_SOURCES[channel]
    except KeyError:
        raise ValueError(f"unknown channel: {channel!r}") from None
    return pl.concat_str([pl.col(column) for column in sources], separator=CHANNEL_SOURCE_SEPARATOR)


def channel_frame(frame: pl.DataFrame, channel: Channel) -> pl.DataFrame:
    r"""`(oracle_id, text)` for every real card that has text in `channel`.

    Whitespace-only text counts as absent. The emptiness check runs on a
    stripped copy so the stored text keeps its original spacing, and the face
    separators ("\n//\n" for prose, " // " for type lines) pass through
    untouched — they are meaningful text for a card carrying two halves
    ([ADR 0002]).
    """
    text = pl.col(TEXT_COLUMN)
    return (
        frame.filter(is_real_card())
        .select(ID_COLUMN, channel_expr(channel).alias(TEXT_COLUMN))
        .filter(text.is_not_null() & (text.str.strip_chars().str.len_chars() > 0))
    )
