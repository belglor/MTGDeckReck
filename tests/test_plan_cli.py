"""Tests for the `just plan` CLI that wires the planner into retrieval.

The planner model and the retrieval half are both faked, so nothing here touches
a weights download or a built index. What is under test is the wiring: which
branch runs, and above all the trap this command exists to get right — the hard
constraints handed to retrieval come from the flags into `Constraints`, never
from the plan the model returns ([ADR 0001]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

import mtg_rag.plan.__main__ as cli
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.filters import Constraints

_THEME = "a spooky graveyard deck that mills itself"
_PLANNED = [
    PlannedQuery(query_text="sacrifice for value", purpose="theme payoff"),
    PlannedQuery(query_text="mana rocks", purpose="ramp"),
]


class _StubPlannerClient:
    """Stands in for `QwenPlannerClient` so no weights load; only has to construct."""


def _fake_plan(request: str, *, format_name: str, client: Any) -> list[PlannedQuery]:
    return _PLANNED


def _stub_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "QwenPlannerClient", _StubPlannerClient)
    monkeypatch.setattr(cli, "plan", _fake_plan)


def test_unknown_format_exits_before_loading_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A format with no template is rejected on a cheap glob, before any weights
    # download — so constructing the planner client here would be a bug.
    def _boom() -> _StubPlannerClient:
        raise AssertionError("the planner model must not load for an unknown format")

    monkeypatch.setattr(cli, "QwenPlannerClient", _boom)

    assert cli.main([_THEME, "--format", "no-such-format"]) == 1
    assert "Unknown format" in capsys.readouterr().err


def test_plan_only_prints_queries_and_never_reads_the_corpus(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_planner(monkeypatch)

    # The corpus load is the witness: --plan-only must not touch it.
    def _no_read(*args: Any, **kwargs: Any) -> pl.DataFrame:
        raise AssertionError("--plan-only must not read the corpus")

    monkeypatch.setattr(pl, "read_parquet", _no_read)

    assert cli.main([_THEME, "--plan-only", "--colors", "B"]) == 0
    out = capsys.readouterr().out
    assert "sacrifice for value" in out
    assert "mana rocks" in out


def test_full_path_hands_retrieval_the_flag_constraints_and_planner_queries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_planner(monkeypatch)

    # A built corpus and index have to exist for the checks to pass; their
    # contents are irrelevant because `read_parquet` and the search are faked.
    (tmp_path / "cards.parquet").write_text("")
    (tmp_path / "vectors").mkdir()

    # `available_formats` reads column names off the frame, so a lone
    # `legal_commander` column is enough to make `commander` a known format.
    frame = pl.DataFrame(schema={"legal_commander": pl.Boolean})

    def _read(*args: Any, **kwargs: Any) -> pl.DataFrame:
        return frame

    monkeypatch.setattr(pl, "read_parquet", _read)

    # Spy on the retrieval hand-off: what main passes here is the whole trap —
    # the queries must be the planner's, the constraints the flags'.
    captured: dict[str, Any] = {}

    def _spy(
        queries: list[PlannedQuery],
        *,
        constraints: Constraints,
        frame: pl.DataFrame,
        vector_dir: Path,
    ) -> None:
        captured["queries"] = queries
        captured["constraints"] = constraints

    monkeypatch.setattr(cli, "_search_and_print", _spy)

    code = cli.main(
        [
            _THEME,
            "--format",
            "commander",
            "--colors",
            "B",
            "--platform",
            "arena",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert captured["queries"] == _PLANNED
    assert captured["constraints"] == Constraints(
        format_name="commander",
        color_identity=frozenset({"B"}),
        platform="arena",
    )


def test_missing_corpus_reports_and_exits_before_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Empty data dir: the existence check fails before the planner loads, with the
    # retrieve CLI's own "run just ingest" message.
    def _boom() -> _StubPlannerClient:
        raise AssertionError("the planner model must not load when the corpus is missing")

    monkeypatch.setattr(cli, "QwenPlannerClient", _boom)

    assert cli.main([_THEME, "--data-dir", str(tmp_path)]) == 1
    assert "just ingest" in capsys.readouterr().err
