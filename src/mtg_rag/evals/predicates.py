"""A predicate kind and its value, as a polars expression.

An eval case names its expected set by a property every card either has or does
not ([ADR 0020]). This is where that name becomes something polars can evaluate.

The same expression is applied twice — over the constrained corpus for a base
rate, and over the returned pool for precision — which is what makes lift a
ratio of like to like. Both call sites take the expression from here, so they
cannot come to disagree about what the case meant.
"""

from __future__ import annotations

import polars as pl


class UnknownPredicateKindError(ValueError):
    """A case named a predicate kind this module cannot build."""


def predicate_expr(kind: str, value: str) -> pl.Expr:
    """The polars predicate for `kind` applied to `value`.

    `keyword` matches Scryfall's `keywords` list **exactly**, capitalisation
    included — the column carries "Madness", not "madness". A case that gets the
    case wrong matches nothing, which surfaces as a zero base rate and is
    refused by `cases.validate_against_corpus` rather than silently scoring 0.

    `oracle_text` is a regex, so it carries its own flags: write `(?i)` for a
    case-insensitive match. A null oracle text is not a match rather than a
    null, matching how `retrieve.filters` guards its own three-valued logic.

    Only the two kinds the golden set actually uses are built. A third would be
    scaffolding for a case that does not exist ([CLAUDE.md]).
    """
    if kind == "keyword":
        return pl.col("keywords").list.contains(value)
    if kind == "oracle_text":
        return pl.col("oracle_text").str.contains(value).fill_null(False)
    raise UnknownPredicateKindError(
        f"unknown predicate kind {kind!r}; available: keyword, oracle_text"
    )
