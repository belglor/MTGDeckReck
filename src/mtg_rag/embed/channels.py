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
from mtg_rag.embed.config import TEXT_COLUMN, Channel


def channel_expr(channel: Channel) -> pl.Expr:
    """The text expression for one channel, before absence filtering.

    The card name is folded into the oracle channel and nowhere else
    ([ADR 0007]): a name is a rules-text-adjacent identifier, and that argument
    does not extend to the type line, which is a controlled vocabulary rather
    than prose. `concat_str` propagates nulls by design, so a vanilla card with
    no oracle text composes to null here and is dropped by `channel_frame`
    rather than embedded as a bare name.
    """
    if channel == "oracle":
        return pl.concat_str([pl.col("name"), pl.col("oracle_text")], separator="\n")
    if channel == "flavor":
        return pl.col("flavor_text")
    if channel == "type":
        return pl.col("type_line")
    raise ValueError(f"unknown channel: {channel!r}")


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
        .select("oracle_id", channel_expr(channel).alias(TEXT_COLUMN))
        .filter(text.is_not_null() & (text.str.strip_chars().str.len_chars() > 0))
    )
