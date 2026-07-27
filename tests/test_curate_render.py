"""Tests for curation's shared display.

Mirrors `retrieve/render.py`'s `print_pool`: pure display, card text hydrated
from `rows` by `oracle_id`. What's under test is the shape ADR 0005 asks
for — grouped by role, a rationale per card, a stable role order, and a
recommendation card missing from `rows` skipped rather than crashing.
"""

from __future__ import annotations

import polars as pl
import pytest

from mtg_rag.curate.recommendation import CuratedCard
from mtg_rag.curate.render import print_recommendation

_ROWS = pl.DataFrame(
    {
        "oracle_id": ["p1", "p2", "e1"],
        "name": ["Splinterfright", "Life from the Loam", "Stinkweed Imp"],
        "mana_cost": ["{3}{G}{G}", "{2}{G}", "{1}{B}"],
        "type_line": ["Creature", "Sorcery", "Creature"],
    }
)


def test_groups_cards_under_role_headings(capsys: pytest.CaptureFixture[str]) -> None:
    recommendation = [
        CuratedCard(oracle_id="p1", role="payoff", rationale="grows from the graveyard"),
        CuratedCard(oracle_id="e1", role="enabler", rationale="self-mills every turn"),
    ]

    print_recommendation(recommendation, _ROWS)

    out = capsys.readouterr().out
    assert "payoff" in out
    assert "enabler" in out
    assert "Splinterfright" in out
    assert "Stinkweed Imp" in out


def test_prints_a_rationale_line_per_card(capsys: pytest.CaptureFixture[str]) -> None:
    recommendation = [
        CuratedCard(oracle_id="p1", role="payoff", rationale="grows from the graveyard"),
    ]

    print_recommendation(recommendation, _ROWS)

    assert "grows from the graveyard" in capsys.readouterr().out


def test_role_order_is_stable_across_runs(capsys: pytest.CaptureFixture[str]) -> None:
    recommendation = [
        CuratedCard(oracle_id="e1", role="enabler", rationale="self-mills every turn"),
        CuratedCard(oracle_id="p2", role="payoff", rationale="recurs lands"),
        CuratedCard(oracle_id="p1", role="payoff", rationale="grows from the graveyard"),
    ]

    first = _captured_output(recommendation, capsys)
    second = _captured_output(recommendation, capsys)

    assert first == second
    assert first.index("enabler") < first.index("payoff")


def test_a_card_missing_from_rows_is_skipped_not_crashed_on(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recommendation = [
        CuratedCard(oracle_id="p1", role="payoff", rationale="grows from the graveyard"),
        CuratedCard(oracle_id="not-a-real-oracle-id", role="payoff", rationale="invented"),
    ]

    print_recommendation(recommendation, _ROWS)

    out = capsys.readouterr().out
    assert "Splinterfright" in out
    assert "invented" not in out


def test_a_role_with_only_missing_cards_does_not_render(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recommendation = [
        CuratedCard(oracle_id="not-a-real-oracle-id", role="ghost role", rationale="invented"),
    ]

    print_recommendation(recommendation, _ROWS)

    assert "ghost role" not in capsys.readouterr().out


def test_an_empty_recommendation_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    print_recommendation([], _ROWS)

    assert capsys.readouterr().out == ""


def _captured_output(recommendation: list[CuratedCard], capsys: pytest.CaptureFixture[str]) -> str:
    print_recommendation(recommendation, _ROWS)
    return capsys.readouterr().out
