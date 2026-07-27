"""Tests for the plan defect checks ([ADR 0026]).

The counting itself is `defects.text`'s, pinned in `test_defects_text.py`. What
is tested here is what these two add: that they read a plan's queries, and that
they carry the `None`-for-empty contract outward rather than swallowing it.

What they are for: #72 records the planner repeating a query within one plan
(`creature sacrifice` three times) and copying the prompt's own worked example
back verbatim (`mana rocks` in 4 of 5 plans in a later sweep). Both are matters
of fact rather than taste, which is what makes them measurable at all
([ADR 0026]) — nothing here judges whether a query is any *good*.
"""

from __future__ import annotations

import pytest

from mtg_rag.defects.plan import duplicate_rate, parroting_rate
from mtg_rag.plan.prompt import EXAMPLE_QUERIES


def test_an_empty_plan_has_no_defined_duplicate_rate() -> None:
    # Undefined, not zero — nothing was measured.
    assert duplicate_rate([]) is None


def test_a_plan_that_repeats_a_query_is_scored_on_its_queries() -> None:
    plan = ["mill", "mill", "graveyard recursion"]

    assert duplicate_rate(plan) == pytest.approx(1 / 3)


def test_an_empty_plan_has_no_defined_parroting_rate() -> None:
    assert parroting_rate([], examples=EXAMPLE_QUERIES) is None


def test_the_shipped_example_queries_are_what_a_parroting_plan_returns() -> None:
    # The end-to-end case the check exists for: a plan made entirely of the
    # prompt's own worked example scores 1.0 against the real constant, so the
    # wiring from prompt to check is proven, not assumed.
    assert parroting_rate(list(EXAMPLE_QUERIES), examples=EXAMPLE_QUERIES) == 1.0


def test_a_plan_of_real_queries_scores_no_parroting_against_the_shipped_examples() -> None:
    plan = ["graveyard recursion", "self-mill enablers", "creature sacrifice payoffs"]

    assert parroting_rate(plan, examples=EXAMPLE_QUERIES) == 0.0
