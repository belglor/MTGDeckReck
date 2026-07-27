"""Tests for printing a sweep ([ADR 0026]).

Pure display over hand-built scores — no model, no sweep. The load-bearing
behaviours are that a run whose plan never validated still gets a row, that the
thematic / mechanical split is reported separately (that comparison is #72's
cross-cutting finding, and one mean would average it away), and that nothing
printed reads as a verdict.
"""

from __future__ import annotations

import pytest

from mtg_rag.defects.render import print_plan_sweep
from mtg_rag.defects.scores import PlanScores, RunScores


def _run(
    theme: str, kind: str, *, duplicate: float | None = 0.0, failed: bool = False
) -> RunScores:
    if failed:
        return RunScores(theme=theme, kind=kind, plan=None, error="not json")
    plan = PlanScores(query_count=4, duplicate_rate=duplicate, parroting_rate=0.25)
    return RunScores(theme=theme, kind=kind, plan=plan)


def test_every_theme_gets_a_row(capsys: pytest.CaptureFixture[str]) -> None:
    print_plan_sweep([_run("graveyard mill", "thematic")], format_name="commander")

    assert "graveyard mill" in capsys.readouterr().out


def test_a_run_whose_plan_never_validated_is_shown_not_skipped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A silently dropped row would make the sweep look cleaner the worse the
    # model got, which is the one failure an instrument must not have.
    print_plan_sweep([_run("broken theme", "thematic", failed=True)], format_name="commander")
    out = capsys.readouterr().out

    assert "broken theme" in out
    assert "1 of 1 did not validate" in out
    assert "did not validate: 1" in out


def test_the_thematic_and_mechanical_means_are_reported_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = [
        _run("evocative one", "thematic", duplicate=1.0),
        _run("mechanical one", "mechanical", duplicate=0.0),
    ]

    print_plan_sweep(runs, format_name="commander")
    out = capsys.readouterr().out

    assert "mean — all" in out
    assert "mean — thematic" in out
    assert "mean — mechanical" in out
    # The split must survive: averaging 1.0 and 0.0 into 0.50 alone would hide
    # exactly the gap #72 recorded.
    assert "1.00" in out
    assert "0.50" in out


def test_a_sweep_where_nothing_validated_reports_no_means(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Undefined, never 0.00 — a zero would read as a clean sweep.
    print_plan_sweep([_run("broken", "thematic", failed=True)], format_name="commander")
    out = capsys.readouterr().out

    assert "0.00" not in out
    assert "—" in out


def test_the_table_is_stamped_with_what_produced_it(capsys: pytest.CaptureFixture[str]) -> None:
    # An unstamped table cannot be compared to a later one (CLAUDE.md).
    print_plan_sweep([_run("graveyard mill", "thematic")], format_name="commander")
    out = capsys.readouterr().out

    assert "model:" in out
    assert "sampling:" in out
    assert "commander" in out
    assert "runs per theme: 1" in out


def test_repeats_of_one_theme_collapse_to_a_single_averaged_row(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = [
        _run("graveyard mill", "thematic", duplicate=0.0),
        _run("graveyard mill", "thematic", duplicate=1.0),
    ]

    print_plan_sweep(runs, format_name="commander")
    out = capsys.readouterr().out

    assert out.count("graveyard mill") == 1
    assert "0.50" in out
    assert "runs per theme: 2" in out


def test_the_widest_spread_within_a_theme_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The noise floor, and the number to read first: a later change smaller
    # than this has not been measured. Two real baseline sweeps moved the mean
    # duplicate rate 0.05 to 0.13 with no code change between them.
    runs = [
        _run("steady theme", "thematic", duplicate=0.10),
        _run("steady theme", "thematic", duplicate=0.20),
        _run("noisy theme", "mechanical", duplicate=0.17),
        _run("noisy theme", "mechanical", duplicate=0.67),
    ]

    print_plan_sweep(runs, format_name="commander")
    out = capsys.readouterr().out

    # 0.67 - 0.17 = 0.50, the widest any single theme showed; not 0.57, which
    # is the spread across *all* runs and would overstate the noise.
    assert "widest spread within one theme: dup 0.50" in out


def test_a_theme_run_only_once_reports_no_spread(capsys: pytest.CaptureFixture[str]) -> None:
    # One sample has no spread; a 0.00 here would read as a stable measurement.
    print_plan_sweep([_run("graveyard mill", "thematic")], format_name="commander")
    out = capsys.readouterr().out

    assert "widest spread within one theme: dup —" in out


def test_a_theme_that_partly_failed_is_averaged_over_what_validated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = [
        _run("graveyard mill", "thematic", duplicate=0.40),
        _run("graveyard mill", "thematic", failed=True),
    ]

    print_plan_sweep(runs, format_name="commander")
    out = capsys.readouterr().out

    assert "0.40" in out
    assert "1 of 2 did not validate" in out


def test_no_verdict_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    # The instrument reports and never gates ([ADR 0020]). The theme is neutral
    # on purpose: the check is over what the *renderer* writes, and a theme
    # containing one of these words would otherwise fail this for no reason.
    print_plan_sweep([_run("graveyard mill", "thematic", duplicate=1.0)], format_name="commander")
    out = capsys.readouterr().out.lower()

    for verdict in ("pass", "fail", "regress", "worse", "bad", "good"):
        assert verdict not in out
