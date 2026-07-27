"""Tests for driving the themes through the planner and scoring them ([ADR 0026]).

A canned `LLMClient` stands in for the model, so nothing here downloads weights.
The arithmetic is pinned in `test_defects_scores`; what is tested here is the
driving — that every theme produces a row, that a theme whose plan never
validates is *recorded* rather than allowed to end the sweep, and that the
committed theme set is what gets run.
"""

from __future__ import annotations

import json

import pytest

from mtg_rag.defects.config import SWEEP_THEMES
from mtg_rag.defects.sweep import run_plan_sweep
from mtg_rag.plan.prompt import EXAMPLE_QUERIES

_THEMES: tuple[tuple[str, str], ...] = (
    ("a spooky graveyard deck that mills itself", "thematic"),
    ("a deck built around +1/+1 counters and proliferate", "mechanical"),
)


def _plan(*queries: str) -> str:
    return json.dumps([{"query_text": q, "purpose": "theme payoff"} for q in queries])


class _FakeClient:
    """A canned `LLMClient`: hands back queued responses and counts its calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self._responses.pop(0)


def test_every_theme_produces_a_row_carrying_its_kind() -> None:
    client = _FakeClient([_plan("graveyard recursion"), _plan("proliferate payoffs")])

    results = run_plan_sweep(format_name="commander", client=client, themes=_THEMES, runs=1)

    assert [(run.theme, run.kind) for run in results] == list(_THEMES)


def test_each_theme_is_planned_once_per_run_and_every_repeat_is_returned() -> None:
    # Repeats are kept rather than averaged in the sweep: sampling noise is
    # large enough that a report which hid the spread would mislead.
    client = _FakeClient([_plan("anything")] * 6)

    results = run_plan_sweep(format_name="commander", client=client, themes=_THEMES, runs=3)

    assert client.calls == 6
    assert [run.theme for run in results] == [_THEMES[0][0]] * 3 + [_THEMES[1][0]] * 3


def test_repeats_of_one_theme_are_scored_independently() -> None:
    # Two runs of the same theme, one clean and one repeating a query — a sweep
    # that reused a score across repeats would report them identical.
    client = _FakeClient([_plan("mill", "recursion"), _plan("mill", "mill")])

    results = run_plan_sweep(format_name="commander", client=client, themes=(_THEMES[0],), runs=2)

    assert [run.plan.duplicate_rate for run in results if run.plan] == [0.0, 0.5]


def test_a_themes_queries_are_what_get_scored() -> None:
    client = _FakeClient([_plan("mill", "mill", EXAMPLE_QUERIES[0]), _plan("proliferate")])

    results = run_plan_sweep(format_name="commander", client=client, themes=_THEMES, runs=1)

    first = results[0].plan
    assert first is not None
    assert first.query_count == 3
    assert first.duplicate_rate == pytest.approx(1 / 3)
    assert first.parroting_rate == pytest.approx(1 / 3)


def test_a_plan_that_never_validates_is_recorded_and_the_sweep_continues() -> None:
    # The planner retries once then raises ([ADR 0022]), so two malformed
    # replies end that theme. The instrument must survive it: an eval that dies
    # on the first bad plan cannot measure how often plans are bad ([ADR 0020]).
    client = _FakeClient(["not json", "still not json", _plan("proliferate payoffs")])

    results = run_plan_sweep(format_name="commander", client=client, themes=_THEMES, runs=1)

    assert len(results) == 2
    assert results[0].plan is None
    assert results[0].error
    assert results[1].plan is not None


def test_a_failed_theme_does_not_consume_a_later_themes_result() -> None:
    # Guards the retry accounting: the planner spends two calls on the failure,
    # so a sweep that mis-tracked responses would score the wrong plan here.
    client = _FakeClient(["not json", "still not json", _plan("a", "a")])

    results = run_plan_sweep(format_name="commander", client=client, themes=_THEMES, runs=1)

    second = results[1].plan
    assert second is not None
    assert second.query_count == 2
    assert second.duplicate_rate == 0.5
    assert client.calls == 3


def test_the_committed_theme_set_is_the_default() -> None:
    client = _FakeClient([_plan("anything")] * len(SWEEP_THEMES))

    results = run_plan_sweep(format_name="commander", client=client, runs=1)

    assert [run.theme for run in results] == [theme for theme, _ in SWEEP_THEMES]
