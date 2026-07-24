"""The four numbers an eval case produces ([ADR 0020]).

Pure arithmetic over a frame — no store, no encoder, no model. The whole point
of separating these from the runner is that the load-bearing rule below can be
tested without any of that.

**The base rate is taken over the constrained corpus, never the whole corpus.**
That is the one mistake here that does not announce itself: dividing by the
whole corpus yields a plausible-looking number and silently makes every lift
under a constraint wrong. `constraint_expr` — the same function retrieval
filters with — is what keeps the denominator and the search in step.

`None` means *undefined*, and is deliberately not 0.0. An empty pool measured
nothing; reporting a zero would put a false data point in a table whose whole
job is to be compared across runs.
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

    Divides by the pool actually returned, never by the requested `k`: a tight
    constraint legitimately returns fewer, and so does any `k` above what three
    channels can surface. Ids the corpus no longer holds are counted in neither
    numerator nor denominator, matching `pool.hydrate`, which drops them.
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
