"""The four numbers an eval case produces ([ADR 0020]).

Pure arithmetic over a frame — no store, no encoder, no model — so the
load-bearing rule below is testable on its own.

**The base rate is taken over the constrained corpus, never the whole corpus.**
Dividing by the whole corpus yields a plausible number that silently makes every
lift under a constraint wrong, so the denominator uses `constraint_expr`, the
same function retrieval filters with.

`None` means *undefined*, never 0.0: an empty pool measured nothing, and a zero
would be a false point in a table meant to be compared across runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.retrieve.filters import Constraints, constraint_expr


def base_rate(frame: pl.DataFrame, constraints: Constraints, predicate: pl.Expr) -> float:
    """The predicate's share of the cards retrieval was **allowed** to return.

    The denominator is the corpus after `constraints` — including the structural
    predicate, which `constraint_expr` applies itself. An empty allowed set
    yields 0.0, which `cases.validate_against_corpus` refuses before any run.
    """
    allowed = frame.filter(constraint_expr(constraints, frame))
    if allowed.height == 0:
        return 0.0
    return allowed.filter(predicate).height / allowed.height


def precision(frame: pl.DataFrame, pool_ids: Sequence[str], predicate: pl.Expr) -> float | None:
    """The predicate's share of the cards retrieval **chose**.

    The ids are looked up in the corpus to get their rows, since the predicate
    tests card fields and a bare id cannot be matched against it. The denominator
    is how many cards actually came back, not the `k` that was requested — a run
    can return fewer — and an empty result has no defined precision, so it is None.
    """
    ids = list(pool_ids)
    if not ids:
        return None
    rows = frame.filter(pl.col(ID_COLUMN).is_in(ids))
    if rows.height == 0:
        return None
    return rows.filter(predicate).height / rows.height


def lift(precision_value: float | None, base_rate_value: float) -> float | None:
    """How much richer the pool is in the property than the corpus was.

    The reported number ([ADR 0020]): precision alone is incomparable between
    cases, between constraint sets, and across corpus refreshes.
    """
    if precision_value is None or base_rate_value <= 0.0:
        return None
    return precision_value / base_rate_value


def retention(lift_value: float | None, reference: float | None) -> float | None:
    """A run's lift against its case's first run — the constraint-interaction number.

    Not clamped to 1.0. A constraint can genuinely *raise* lift by narrowing the
    corpus toward the theme, and a value below 1.0 is not automatically a
    regression: it also happens when the constrained corpus is already dense in
    the property, leaving less headroom to enrich ([ADR 0020]).
    """
    if lift_value is None or reference is None or reference <= 0.0:
        return None
    return lift_value / reference
