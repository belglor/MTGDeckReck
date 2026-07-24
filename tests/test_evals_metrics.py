"""Tests for the eval metrics: base rate, precision, lift, retention.

Frames are built through `normalize_card` + `build_frame` so the schema under
test is the one ingestion writes. Nothing here touches a model, the store, or
the network — these are arithmetic over a frame.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from mtg_rag.evals.metrics import base_rate, lift, precision, retention
from mtg_rag.evals.predicates import predicate_expr
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.retrieve.filters import Constraints


def _card(
    name: str,
    *,
    keywords: list[str] | None = None,
    oracle_text: str | None = None,
    colors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "oracle_id": f"id-{name}",
        "name": name,
        "oracle_text": oracle_text,
        "type_line": "Creature — Test",
        "keywords": keywords or [],
        "color_identity": colors or [],
        "layout": "normal",
        "set_type": "expansion",
        "released_at": "2020-01-01",
        "games": ["paper"],
        "legalities": {"commander": "legal"},
    }


#: Four white cards, one with Madness; six black cards, three with Madness.
#: The whole-frame rate is 4/10 = 40%, the mono-white-fittable rate is 1/4 =
#: 25%. Every base-rate test below turns on that difference.
CARDS = [
    _card("white-madness", keywords=["Madness"], colors=["W"]),
    _card("white-plain-1", colors=["W"]),
    _card("white-plain-2", colors=["W"]),
    _card("white-plain-3", colors=["W"]),
    _card("black-madness-1", keywords=["Madness"], colors=["B"]),
    _card("black-madness-2", keywords=["Madness"], colors=["B"]),
    _card("black-madness-3", keywords=["Madness"], colors=["B"]),
    _card("black-plain-1", colors=["B"]),
    _card("black-plain-2", colors=["B"]),
    _card("black-plain-3", colors=["B"]),
]


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    return build_frame([normalize_card(card) for card in CARDS])


@pytest.fixture(scope="module")
def madness() -> pl.Expr:
    return predicate_expr("keyword", "Madness")


# --- base rate -------------------------------------------------------------


def test_base_rate_is_taken_over_the_constrained_corpus(
    corpus: pl.DataFrame, madness: pl.Expr
) -> None:
    """The load-bearing test of the whole harness ([ADR 0020]).

    Dividing by the whole corpus instead produces 0.40 here — a number that
    looks entirely plausible and silently makes every lift under a colour
    constraint wrong.
    """
    white = Constraints("commander", frozenset({"W"}))
    assert base_rate(corpus, white, madness) == pytest.approx(0.25)


def test_base_rate_unconstrained_covers_the_whole_allowed_corpus(
    corpus: pl.DataFrame, madness: pl.Expr
) -> None:
    assert base_rate(corpus, Constraints("commander"), madness) == pytest.approx(0.40)


def test_base_rate_reflects_a_tighter_constraint(corpus: pl.DataFrame, madness: pl.Expr) -> None:
    black = Constraints("commander", frozenset({"B"}))
    assert base_rate(corpus, black, madness) == pytest.approx(0.50)


# --- precision -------------------------------------------------------------


def test_precision_counts_predicate_matches_in_the_pool(
    corpus: pl.DataFrame, madness: pl.Expr
) -> None:
    pool = ["id-white-madness", "id-black-madness-1", "id-white-plain-1", "id-black-plain-1"]
    assert precision(corpus, pool, madness) == pytest.approx(0.5)


def test_precision_divides_by_actual_pool_size_not_k(
    corpus: pl.DataFrame, madness: pl.Expr
) -> None:
    """A tight constraint returns fewer than `k`; `k` must not be the divisor."""
    pool = ["id-white-madness", "id-white-plain-1"]
    assert precision(corpus, pool, madness) == pytest.approx(0.5)


def test_empty_pool_has_undefined_precision(corpus: pl.DataFrame, madness: pl.Expr) -> None:
    """Undefined, not zero — nothing was measured, so nothing is reported."""
    assert precision(corpus, [], madness) is None


def test_precision_ignores_ids_absent_from_the_corpus(
    corpus: pl.DataFrame, madness: pl.Expr
) -> None:
    """Dropped from numerator and denominator alike, as `pool.hydrate` drops them.

    Counting a vanished card as a miss would report a retrieval failure for a
    card that left the corpus — a fact about ingestion, not about ranking.
    """
    pool = ["id-white-madness", "id-does-not-exist"]
    assert precision(corpus, pool, madness) == pytest.approx(1.0)


# --- lift and retention ----------------------------------------------------


def test_lift_is_precision_over_base_rate() -> None:
    assert lift(0.5, 0.25) == pytest.approx(2.0)


def test_lift_of_an_undefined_precision_is_undefined() -> None:
    assert lift(None, 0.25) is None


def test_lift_of_a_zero_base_rate_is_undefined() -> None:
    """Guarded even though a zero base rate is refused at validation."""
    assert lift(0.5, 0.0) is None


def test_retention_compares_a_run_against_its_reference() -> None:
    assert retention(4.0, 5.0) == pytest.approx(0.8)


def test_retention_of_the_reference_run_is_one() -> None:
    assert retention(5.0, 5.0) == pytest.approx(1.0)


def test_retention_above_one_is_representable() -> None:
    """A constraint can raise lift; the metric must not clamp ([ADR 0020])."""
    assert retention(6.0, 3.0) == pytest.approx(2.0)


def test_retention_is_undefined_when_either_side_is() -> None:
    assert retention(None, 5.0) is None
    assert retention(5.0, None) is None
    assert retention(5.0, 0.0) is None
