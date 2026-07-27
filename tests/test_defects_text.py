"""Tests for the two defect rates shared by both model stages ([ADR 0026]).

`repeat_rate` and `copied_rate` count the same thing over a plan's queries and
a recommendation's rationales, so their semantics are pinned once here and the
stage modules are left to test that they read the right field.

`None` means undefined, never 0.0: an empty sequence exhibited no defect and
measured nothing, and a zero would be a false point in a table compared across
runs (the rule `evals/metrics.py` already follows).
"""

from __future__ import annotations

import pytest

from mtg_rag.defects.text import copied_rate, normalize, repeat_rate

# --- repeat_rate -----------------------------------------------------------


def test_an_empty_sequence_has_no_defined_repeat_rate() -> None:
    assert repeat_rate([]) is None


def test_nothing_repeated_scores_zero() -> None:
    assert repeat_rate(["graveyard recursion", "self-mill enablers", "mana rocks"]) == 0.0


def test_a_lone_entry_cannot_repeat_itself() -> None:
    assert repeat_rate(["graveyard recursion"]) == 0.0


def test_one_repeat_of_two_is_half() -> None:
    # The second occurrence is the defect; the first is the entry.
    assert repeat_rate(["mill", "mill"]) == 0.5


def test_every_occurrence_beyond_the_first_counts() -> None:
    # #72's aristocrats run emitted `creature sacrifice` three times: two of the
    # three slots were wasted, not one.
    assert repeat_rate(["creature sacrifice"] * 3) == pytest.approx(2 / 3)


def test_repeats_are_counted_per_distinct_entry() -> None:
    assert repeat_rate(["a", "a", "b", "b"]) == 0.5


def test_case_and_surrounding_whitespace_do_not_make_an_entry_distinct() -> None:
    assert repeat_rate(["Mill", " mill "]) == 0.5


def test_internal_wording_still_makes_an_entry_distinct() -> None:
    # Normalization is deliberately shallow: only case and outer whitespace.
    assert repeat_rate(["mill", "self-mill"]) == 0.0


# --- copied_rate -----------------------------------------------------------


def test_an_empty_sequence_has_no_defined_copied_rate() -> None:
    assert copied_rate([], ("mana rocks",)) is None


def test_copying_nothing_scores_zero() -> None:
    assert copied_rate(["graveyard recursion"], ("mana rocks",)) == 0.0


def test_a_copied_entry_is_counted() -> None:
    assert copied_rate(["mana rocks", "graveyard recursion"], ("mana rocks",)) == 0.5


def test_copying_wholesale_scores_one() -> None:
    examples = ("mana rocks", "sacrifice for value")

    assert copied_rate(list(examples), examples) == 1.0


def test_copying_ignores_case_and_surrounding_whitespace() -> None:
    assert copied_rate([" Mana Rocks "], ("mana rocks",)) == 1.0


def test_merely_containing_an_example_is_not_copying() -> None:
    # The defect is handing the example back, not using one of its words. A
    # substring rule would flag a legitimately more specific entry and inflate
    # the number exactly where the model did better.
    assert copied_rate(["mana rocks that untap"], ("mana rocks",)) == 0.0


def test_each_copied_entry_counts_even_when_repeated() -> None:
    # Repetition is `repeat_rate`'s business; this counts how much was copied.
    assert copied_rate(["mana rocks", "mana rocks"], ("mana rocks",)) == 1.0


# --- normalize -------------------------------------------------------------


def test_normalize_folds_case_and_strips_the_outside_only() -> None:
    # Pinned because both rates rest on it: inner spacing and punctuation are
    # left alone on purpose, so "self mill" and "self-mill" stay distinct.
    assert normalize("  Mana  Rocks ") == "mana  rocks"
