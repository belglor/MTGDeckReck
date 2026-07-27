"""Tests for the `just plan` CLI that runs Plan → Retrieve → Curate end to end.

Both models and the retrieval half are faked, so nothing here touches a weights
download or a built index. What is under test is the wiring: which branch runs,
that the curated recommendation is drawn from the retrieved pool, and above all
the trap this command exists to get right — the hard constraints handed to
retrieval come from the flags into `Constraints`, never from the plan the model
returns ([ADR 0001]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import mtg_rag.curate.config as curate_config
import mtg_rag.plan.__main__ as cli
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.retrieve.fusion import Candidate, Source

_THEME = "a spooky graveyard deck that mills itself"
_PLANNED = [
    PlannedQuery(query_text="sacrifice for value", purpose="theme payoff"),
    PlannedQuery(query_text="mana rocks", purpose="ramp"),
]

# A two-card pool and the rows `hydrate` would return for it, standing in for
# the retrieval half. `id-mill` was found twice by the same purpose, which is
# what makes the de-duplication in `_curation_cards` visible.
_POOL = [
    Candidate(
        oracle_id="id-mill",
        score=0.5,
        sources=(
            Source(purpose="theme payoff", channel="oracle", rank=0, distance=0.1),
            Source(purpose="theme payoff", channel="flavor", rank=2, distance=0.3),
            Source(purpose="self-mill enabler", channel="type", rank=4, distance=0.4),
        ),
    ),
    Candidate(
        oracle_id="id-rock",
        score=0.4,
        sources=(Source(purpose="ramp", channel="type", rank=1, distance=0.2),),
    ),
]
_ROWS = pl.DataFrame(
    {
        "oracle_id": ["id-mill", "id-rock"],
        "name": ["Mesmeric Orb", "Sol Ring"],
        "mana_cost": ["{2}", None],
        "type_line": ["Artifact", "Artifact"],
        "oracle_text": [
            "Whenever a permanent becomes untapped, its controller mills a card.",
            None,
        ],
        "flavor_text": [None, "The perfect tool."],
    }
)

_RECOMMENDATION = json.dumps(
    [
        {
            "oracle_id": "id-mill",
            "role": "theme payoff",
            "rationale": "Mills the whole table, filling the graveyard the deck feeds on.",
        },
        {
            "oracle_id": "id-rock",
            "role": "ramp",
            "rationale": "Two colorless mana a turn, so the graveyard payoffs land early.",
        },
    ]
)


class _StubPlannerClient:
    """Stands in for `QwenChatClient` so no weights load; only has to construct."""


class _FakeChatClient:
    """An `LLMClient` that replies with canned text and counts the asks.

    Curation reuses whatever client the planner loaded, so this doubles as the
    witness that the CLI does not load a second copy of the model.
    """

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.asks: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.asks.append(user)
        # The last reply repeats, so a retry re-asking gets the same answer.
        return self.replies[min(len(self.asks) - 1, len(self.replies) - 1)]


def _fake_plan(request: str, *, format_name: str, client: Any) -> list[PlannedQuery]:
    return _PLANNED


def _stub_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "QwenChatClient", _StubPlannerClient)
    monkeypatch.setattr(cli, "plan", _fake_plan)


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, client: _FakeChatClient) -> None:
    """Fake both models and the retrieval half, leaving curation's wiring real."""

    def _client() -> _FakeChatClient:
        return client

    def _found(*args: Any, **kwargs: Any) -> tuple[list[Candidate], pl.DataFrame]:
        return _POOL, _ROWS

    monkeypatch.setattr(cli, "QwenChatClient", _client)
    monkeypatch.setattr(cli, "plan", _fake_plan)
    monkeypatch.setattr(cli, "_search_and_print", _found)


def _built_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A data dir that passes the CLI's existence and format checks.

    The files' contents are irrelevant: `read_parquet` is faked, and the lone
    `legal_commander` column is all `available_formats` reads to make
    `commander` a known format.
    """
    (tmp_path / "cards.parquet").write_text("")
    (tmp_path / "vectors").mkdir()
    frame = pl.DataFrame(schema={"legal_commander": pl.Boolean})

    def _read(*args: Any, **kwargs: Any) -> pl.DataFrame:
        return frame

    monkeypatch.setattr(pl, "read_parquet", _read)
    return tmp_path


def test_unknown_format_exits_before_loading_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A format with no template is rejected on a cheap glob, before any weights
    # download — so constructing the planner client here would be a bug.
    def _boom() -> _StubPlannerClient:
        raise AssertionError("the planner model must not load for an unknown format")

    monkeypatch.setattr(cli, "QwenChatClient", _boom)

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
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    # Spy on the retrieval hand-off: what main passes here is the whole trap —
    # the queries must be the planner's, the constraints the flags'.
    captured: dict[str, Any] = {}

    def _spy(
        queries: list[PlannedQuery],
        *,
        constraints: Constraints,
        frame: pl.DataFrame,
        vector_dir: Path,
    ) -> tuple[list[Candidate], pl.DataFrame]:
        captured["queries"] = queries
        captured["constraints"] = constraints
        return _POOL, _ROWS

    def _skip_curation(*args: Any, **kwargs: Any) -> int:
        return 0

    monkeypatch.setattr(cli, "_search_and_print", _spy)
    monkeypatch.setattr(cli, "_curate_and_print", _skip_curation)

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
            str(data_dir),
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

    monkeypatch.setattr(cli, "QwenChatClient", _boom)

    assert cli.main([_THEME, "--data-dir", str(tmp_path)]) == 1
    assert "just ingest" in capsys.readouterr().err


def test_full_path_prints_a_role_grouped_recommendation_from_the_pool(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    client = _FakeChatClient(_RECOMMENDATION)
    _stub_pipeline(monkeypatch, client)
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    assert cli.main([_THEME, "--data-dir", str(data_dir)]) == 0

    out = capsys.readouterr().out
    assert "== theme payoff ==" in out
    assert "== ramp ==" in out
    assert "Mesmeric Orb" in out
    assert "Sol Ring" in out
    assert "filling the graveyard the deck feeds on" in out

    # One ask, on the client the planner already loaded — curation must not
    # stand up a second copy of the model.
    assert len(client.asks) == 1


def test_full_path_shows_curation_the_pool_paired_with_its_purposes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeChatClient(_RECOMMENDATION)
    _stub_pipeline(monkeypatch, client)
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    assert cli.main([_THEME, "--data-dir", str(data_dir)]) == 0
    asked = client.asks[0]

    # What reaches the model is the retrieved pool and nothing else, so a card
    # the filters excluded cannot be recommended ([ADR 0001]).
    assert asked.count("oracle_id:") == len(_POOL)
    assert _THEME in asked

    # Each card carries the purposes of the searches that found it — the
    # starting hypothesis for its role ([ADR 0005]) — de-duplicated but in
    # source order. Null corpus fields drop their line rather than printing
    # blank: Sol Ring's row has no oracle text and no mana cost here.
    assert "found for: theme payoff, self-mill enabler" in asked
    assert asked.count("found for: ") == len(_POOL)
    assert "mana cost: {2}" in asked
    assert asked.count("mana cost: ") == 1
    assert asked.count("text: ") == 1


def test_curation_sees_only_the_top_of_a_pool_that_exceeds_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The whole pool does not fit in the model's context on the target GPU, so
    # curation is shown the top `CURATION_POOL_SIZE` of it.
    monkeypatch.setattr(curate_config, "CURATION_POOL_SIZE", 1)
    client = _FakeChatClient(json.dumps([]))
    _stub_pipeline(monkeypatch, client)
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    assert cli.main([_THEME, "--data-dir", str(data_dir)]) == 0

    asked = client.asks[0]
    assert asked.count("oracle_id:") == 1
    assert "Mesmeric Orb" in asked
    assert "Sol Ring" not in asked


def test_pool_only_stops_before_curation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    client = _FakeChatClient(_RECOMMENDATION)
    _stub_pipeline(monkeypatch, client)
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    def _boom(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("--pool-only must not curate")

    monkeypatch.setattr(cli, "_curate_and_print", _boom)

    assert cli.main([_THEME, "--pool-only", "--data-dir", str(data_dir)]) == 0
    assert "==" not in capsys.readouterr().out


def test_empty_pool_never_calls_curation_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An unsatisfiable request is honest output, not an error: the "no
    # candidates" line already stood, and there is nothing to curate.
    _stub_planner(monkeypatch)
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    def _nothing_found(*args: Any, **kwargs: Any) -> None:
        return None

    def _boom(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("an empty pool must not reach curation")

    monkeypatch.setattr(cli, "_search_and_print", _nothing_found)
    monkeypatch.setattr(cli, "_curate_and_print", _boom)

    assert cli.main([_THEME, "--data-dir", str(data_dir)]) == 0


def test_malformed_curation_reports_one_line_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Curation re-asks once and then raises ([ADR 0024]); the CLI turns that into
    # one clear line rather than letting the traceback out.
    client = _FakeChatClient("not json at all")
    _stub_pipeline(monkeypatch, client)
    data_dir = _built_data_dir(monkeypatch, tmp_path)

    assert cli.main([_THEME, "--data-dir", str(data_dir)]) == 1

    captured = capsys.readouterr()
    assert len(client.asks) == 2
    assert len(captured.err.strip().splitlines()) == 1
    assert "recommendation" in captured.err
    assert "== " not in captured.out
