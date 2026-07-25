"""What a predicate is, and how it becomes a polars expression.

An eval case names its expected set by a property every card has or lacks
([ADR 0020]); here that name becomes a polars expression. The same expression is
applied twice — over the constrained corpus for a base rate, over the pool for
precision — so both call sites take it from here and can't disagree about what
the case meant.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mtg_rag.evals.config import PredicateKind


class UnknownPredicateKindError(ValueError):
    """A case named a predicate kind this module cannot build."""


@dataclass(frozen=True, slots=True)
class Predicate:
    """The property a case expects the pool to be enriched for."""

    kind: PredicateKind
    value: str

    def expr(self) -> pl.Expr:
        return predicate_expr(self.kind, self.value)


def predicate_expr(kind: str, value: str) -> pl.Expr:
    """The polars predicate for `kind` applied to `value`.

    `keyword` matches Scryfall's `keywords` list **exactly**, capitalisation
    included — the column carries "Madness", not "madness". A case that gets the
    case wrong matches nothing, which surfaces as a zero base rate and is
    refused by `cases.validate_against_corpus` rather than silently scoring 0.

    `oracle_text` and `type_line` are regexes, so they carry their own flags:
    write `(?i)` for a case-insensitive match. A null column value is not a match
    rather than a null, matching how `retrieve.filters` guards its own
    three-valued logic. `type_line` reads creature type and card type off the
    printed type line — a graveyard-creature case matches on the subtypes.

    Only the kinds a golden-set case actually uses are built. A new kind is added
    when a case needs it, never for symmetry ([CLAUDE.md]).
    """
    if kind == "keyword":
        return pl.col("keywords").list.contains(value)
    if kind == "oracle_text":
        return pl.col("oracle_text").str.contains(value).fill_null(False)
    if kind == "type_line":
        return pl.col("type_line").str.contains(value).fill_null(False)
    raise UnknownPredicateKindError(
        f"unknown predicate kind {kind!r}; available: keyword, oracle_text, type_line"
    )
