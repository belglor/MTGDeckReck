"""Tests for the plan defect checks ([ADR 0026]).

Both checks are pure functions over query texts — no model, no corpus — so
every case here is a literal list in, a number out.

What they are for: #72 records the planner repeating a query within one plan
(`creature sacrifice` three times) and copying the prompt's own worked example
back verbatim (`mana rocks` in 4 of 5 plans in a later sweep). Both are matters
of fact rather than taste, which is what makes them measurable at all
([ADR 0026]) — nothing here judges whether a query is any *good*.

`None` means undefined, never 0.0: an empty plan exhibited neither defect and
measured nothing, and a zero would be a false point in a table compared across
runs (the rule `evals/metrics.py` already follows).
"""

from __future__ import annotations

import pytest

from mtg_rag.defects.plan import duplicate_rate, parroting_rate
from mtg_rag.plan.prompt import EXAMPLE_QUERIES

# --- duplicate_rate --------------------------------------------------------


def test_an_empty_plan_has_no_defined_duplicate_rate() -> None:
    # Undefined, not zero — nothing was measured.
    assert duplicate_rate([]) is None


def test_a_plan_with_no_repeats_scores_zero() -> None:
    assert duplicate_rate(["graveyard recursion", "self-mill enablers", "mana rocks"]) == 0.0


def test_a_single_query_cannot_repeat_itself() -> None:
    assert duplicate_rate(["graveyard recursion"]) == 0.0


def test_one_repeat_of_two_queries_is_half() -> None:
    # The second occurrence is the defect; the first is the query.
    assert duplicate_rate(["mill", "mill"]) == 0.5


def test_every_extra_occurrence_counts() -> None:
    # #72's aristocrats run emitted `creature sacrifice` three times: two of the
    # three slots were wasted, not one.
    assert duplicate_rate(["creature sacrifice"] * 3) == pytest.approx(2 / 3)


def test_repeats_are_counted_per_distinct_query() -> None:
    assert duplicate_rate(["a", "a", "b", "b"]) == 0.5


def test_case_and_surrounding_whitespace_do_not_make_a_query_distinct() -> None:
    # The planner writing "Mill" and " mill " has still asked twice; retrieval
    # would embed them to near-identical vectors and fuse the same cards.
    assert duplicate_rate(["Mill", " mill "]) == 0.5


def test_internal_wording_still_makes_a_query_distinct() -> None:
    # Normalization is deliberately shallow: only case and outer whitespace.
    # Two differently-worded queries are two queries, however similar.
    assert duplicate_rate(["mill", "self-mill"]) == 0.0


# --- parroting_rate --------------------------------------------------------


def test_an_empty_plan_has_no_defined_parroting_rate() -> None:
    assert parroting_rate([], examples=EXAMPLE_QUERIES) is None


def test_a_plan_that_copies_nothing_scores_zero() -> None:
    assert parroting_rate(["graveyard recursion"], examples=("mana rocks",)) == 0.0


def test_a_query_copied_from_the_example_is_counted() -> None:
    assert parroting_rate(["mana rocks", "graveyard recursion"], examples=("mana rocks",)) == 0.5


def test_a_plan_copied_wholesale_scores_one() -> None:
    assert parroting_rate(["mana rocks", "sacrifice for value"], examples=EXAMPLE_QUERIES) == 1.0


def test_matching_ignores_case_and_surrounding_whitespace() -> None:
    assert parroting_rate([" Mana Rocks "], examples=("mana rocks",)) == 1.0


def test_a_query_that_merely_contains_an_example_is_not_parroting() -> None:
    # The defect is copying the example back, not using one of its words. A
    # substring rule would flag a legitimately more specific query and inflate
    # the number exactly where the planner did better.
    assert parroting_rate(["mana rocks that untap"], examples=("mana rocks",)) == 0.0


def test_each_copied_query_counts_once_even_when_repeated() -> None:
    # Repetition is `duplicate_rate`'s business; this counts how much of the
    # plan is copied, so both slots are parroted here.
    assert parroting_rate(["mana rocks", "mana rocks"], examples=("mana rocks",)) == 1.0
