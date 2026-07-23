"""Turn the user's hard constraints into the `oracle_id` allowlist a search may
return.

Format legality, color identity, and platform are deterministic metadata filters
specified by the user, never inferred from the query or policed by an LLM
([ADR 0001]). They are evaluated against the corpus parquet — the single source
of truth — and the surviving ids constrain the vector search ([ADR 0010]). No
vector or store is touched here.

Each constraint is a `pl.Expr` so they compose into one `.filter()` pass,
alongside the structural predicate `is_real_card` ([ADR 0013]) — which this
applies itself rather than trusting every caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mtg_rag.corpus import is_real_card
from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.ingest.config import PLATFORMS
from mtg_rag.retrieve.config import (
    DEFAULT_PLATFORM,
    LEGALITY_COLUMN_PREFIX,
    PLAYABLE_LEGALITIES,
)


@dataclass(frozen=True, slots=True)
class Constraints:
    """What the user picked in the UI ([ADR 0001]).

    `color_identity=None` is unconstrained; `frozenset()` is colorless-only.
    These are different answers — every deck versus a colorless commander's —
    so the distinction is in the type rather than a sentinel.
    """

    format_name: str  # not `format` — that shadows the builtin
    color_identity: frozenset[str] | None = None
    platform: str = DEFAULT_PLATFORM


def available_formats(frame: pl.DataFrame) -> tuple[str, ...]:
    """The formats the corpus carries, read off its `legal_<format>` columns."""
    return tuple(
        column.removeprefix(LEGALITY_COLUMN_PREFIX)
        for column in frame.columns
        if column.startswith(LEGALITY_COLUMN_PREFIX)
    )


def legality_expr(format_name: str, frame: pl.DataFrame) -> pl.Expr:
    """Cards playable in `format_name`. Raises on an unknown format.

    The user picks a format in a UI, so a typo should say what was available
    rather than silently return nothing. A null legality counts as not-playable,
    guarded with `fill_null(False)` — polars' three-valued `filter` would drop
    the row anyway, but for the wrong reason and reversibly.
    """
    formats = available_formats(frame)
    if format_name not in formats:
        available = ", ".join(sorted(formats))
        raise ValueError(f"unknown format {format_name!r}; available: {available}")
    column = f"{LEGALITY_COLUMN_PREFIX}{format_name}"
    return pl.col(column).is_in(PLAYABLE_LEGALITIES).fill_null(False)


def color_identity_expr(identity: frozenset[str]) -> pl.Expr:
    """Cards whose color identity fits within `identity`.

    The subset test: a card is allowed when it uses no color the deck lacks.
    A colorless card (empty identity) fits every deck. User input is uppercased
    to match the corpus's WUBRG.
    """
    allowed = [color.upper() for color in identity]
    return pl.col("color_identity").list.set_difference(allowed).list.len() == 0


def platform_expr(platform: str) -> pl.Expr:
    """Cards available on `platform`. Raises on an unknown one."""
    if platform not in PLATFORMS:
        available = ", ".join(PLATFORMS)
        raise ValueError(f"unknown platform {platform!r}; available: {available}")
    return pl.col("platforms").list.contains(platform)


def constraint_expr(constraints: Constraints, frame: pl.DataFrame) -> pl.Expr:
    """Every constraint, plus the structural predicate, as one expression."""
    expr = (
        is_real_card()
        & legality_expr(constraints.format_name, frame)
        & platform_expr(constraints.platform)
    )
    if constraints.color_identity is not None:
        expr = expr & color_identity_expr(constraints.color_identity)
    return expr


def allowed_ids(frame: pl.DataFrame, constraints: Constraints) -> list[str]:
    """The sorted `oracle_id`s a search under `constraints` may return.

    Sorted and deduplicated so the pool the store builds downstream is
    reproducible. An empty result is a valid answer, not an error.
    """
    return (
        frame.filter(constraint_expr(constraints, frame))
        .get_column(ID_COLUMN)
        .unique()
        .sort()
        .to_list()
    )
