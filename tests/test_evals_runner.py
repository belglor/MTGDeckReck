"""Tests for the eval runner: what a run produces, and what it refuses to do.

The runner is handed a `Retriever` ([ADR 0020]), so these tests supply a
stand-in that returns canned ids — no model, no store, no Chroma. Retrieval has
its own tests; here the pool is an input, which lets every metric be asserted to
an exact value and keeps the suite off the vector store entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
import pytest

from mtg_rag.embed.config import CHANNELS
from mtg_rag.embed.index import VectorIndex
from mtg_rag.evals.cases import EvalCase
from mtg_rag.evals.predicates import Predicate
from mtg_rag.evals.runner import CaseResult, Report, Retriever, Run, run_case, run_cases
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.filters import Constraints


def _card(
    name: str, *, keywords: list[str] | None = None, colors: list[str] | None = None
) -> dict[str, Any]:
    return {
        "oracle_id": f"id-{name}",
        "name": name,
        "oracle_text": None,
        "type_line": "Creature — Test",
        "keywords": keywords or [],
        "color_identity": colors or [],
        "layout": "normal",
        "set_type": "expansion",
        "released_at": "2020-01-01",
        "games": ["paper"],
        "legalities": {"commander": "legal"},
    }


#: Cycling density differs by colour on purpose, so a base rate that follows the
#: constraint can be told apart from one that does not: 3/6 over the whole set,
#: 1/3 among the white-fittable cards, 2/3 among the black.
CARDS = [
    _card("wc1", keywords=["Cycling"], colors=["W"]),
    _card("wp1", colors=["W"]),
    _card("wp2", colors=["W"]),
    _card("bc1", keywords=["Cycling"], colors=["B"]),
    _card("bc2", keywords=["Cycling"], colors=["B"]),
    _card("bp1", colors=["B"]),
]


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    return build_frame([normalize_card(card) for card in CARDS])


def _case(constraints: tuple[Constraints, ...], *, keyword: str = "Cycling") -> EvalCase:
    return EvalCase(
        id="cycling",
        query="cycling",
        rationale="Scryfall's keywords column is the ground truth.",
        predicate=Predicate(kind="keyword", value=keyword),
        constraints=constraints,
    )


def _fixed(*ids: str) -> Retriever:
    """A retriever that returns the same ids under every constraint set."""

    def retriever(queries: Sequence[PlannedQuery], constraints: Constraints) -> list[str]:
        return list(ids)

    return retriever


def _per_color(pools: dict[frozenset[str] | None, list[str]]) -> Retriever:
    """A retriever whose pool depends on the constraint's colour identity."""

    def retriever(queries: Sequence[PlannedQuery], constraints: Constraints) -> list[str]:
        return pools[constraints.color_identity]

    return retriever


INDEX = VectorIndex(
    model_id="test/model",
    dim=2,
    corpus_updated_at="2026-07-22T21:12:36.682+00:00",
    corpus_row_count=6,
    channel_counts={"oracle": 6, "flavor": 0, "type": 6},
    embedded_at="2026-07-23T20:40:10.699755+00:00",
)


# --- what a run produces ---------------------------------------------------


def test_one_run_per_constraint_set_in_file_order(corpus: pl.DataFrame) -> None:
    case = _case((Constraints("commander"), Constraints("commander", frozenset({"W"}))))
    result = run_case(case, frame=corpus, retriever=_fixed("id-wc1"))
    assert [run.constraints.color_identity for run in result.runs] == [None, frozenset({"W"})]


def test_the_first_constraint_set_is_the_retention_reference(corpus: pl.DataFrame) -> None:
    result = run_case(_case((Constraints("commander"),)), frame=corpus, retriever=_fixed("id-wc1"))
    assert result.runs[0].retention == pytest.approx(1.0)


def test_retention_is_exactly_lift_over_the_reference(corpus: pl.DataFrame) -> None:
    # commander: precision 2/2 over base 3/6  -> lift 2.0
    # white:     precision 1/1 over base 1/3  -> lift 3.0  -> retention 1.5
    retriever = _per_color({None: ["id-wc1", "id-bc1"], frozenset({"W"}): ["id-wc1"]})
    ref, tighter = run_case(
        _case((Constraints("commander"), Constraints("commander", frozenset({"W"})))),
        frame=corpus,
        retriever=retriever,
    ).runs
    assert ref.lift == pytest.approx(2.0)
    assert tighter.lift == pytest.approx(3.0)
    assert tighter.retention == pytest.approx(1.5)


def test_base_rate_follows_the_constraint(corpus: pl.DataFrame) -> None:
    """The denominator must track the constraint, or lift means nothing."""
    ref, tighter = run_case(
        _case((Constraints("commander"), Constraints("commander", frozenset({"W"})))),
        frame=corpus,
        retriever=_fixed("id-wc1"),
    ).runs
    assert ref.base_rate == pytest.approx(0.5)
    assert tighter.base_rate == pytest.approx(1 / 3)


def test_pool_size_is_the_number_of_ids_returned(corpus: pl.DataFrame) -> None:
    result = run_case(
        _case((Constraints("commander"),)),
        frame=corpus,
        retriever=_fixed("id-wc1", "id-bc1", "id-wp1"),
    )
    assert result.runs[0].pool_size == 3


# --- what a run refuses to do ----------------------------------------------


def test_a_poor_metric_does_not_fail_the_run(corpus: pl.DataFrame) -> None:
    """A number the reader dislikes is the output, not an error ([ADR 0011])."""
    case = _case((Constraints("commander"),), keyword="Fuse")
    result = run_case(case, frame=corpus, retriever=_fixed("id-wp1"))
    assert result.runs[0].precision == 0.0


def test_an_empty_pool_reports_undefined_rather_than_raising(corpus: pl.DataFrame) -> None:
    """Retrieval can legitimately return nothing; that is honest output, not an error."""
    result = run_case(_case((Constraints("commander"),)), frame=corpus, retriever=_fixed())
    run = result.runs[0]
    assert run.pool_size == 0
    assert run.precision is None
    assert run.lift is None


def test_run_cases_returns_one_result_per_case(corpus: pl.DataFrame) -> None:
    """The only thing `run_cases` does that `run_case` does not."""
    report = run_cases(
        [_case((Constraints("commander"),)), _case((Constraints("commander"),), keyword="Fuse")],
        frame=corpus,
        retriever=_fixed("id-wc1"),
        index=INDEX,
        k=10,
        channels=CHANNELS,
    )
    assert len(report.results) == 2
    assert report.index is INDEX and report.k == 10


# --- the report as JSON ----------------------------------------------------
# Built as a literal rather than run. Every value these assert was known before
# any retrieval happened, so going through the store would only make them slower
# and vaguer — and a literal can pin exact values, which a live run cannot.

REPORT = Report(
    index=INDEX,
    k=10,
    channels=CHANNELS,
    results=(
        CaseResult(
            case=_case(
                (
                    Constraints("commander"),
                    Constraints("commander", frozenset({"W"})),
                    Constraints("commander", frozenset()),
                )
            ),
            runs=(
                Run(
                    constraints=Constraints("commander"),
                    pool_size=25,
                    base_rate=0.13954,
                    precision=0.72,
                    lift=5.160105,
                    retention=1.0,
                ),
                Run(
                    constraints=Constraints("commander", frozenset({"W"})),
                    pool_size=25,
                    base_rate=0.100628,
                    precision=0.52,
                    lift=5.167543160690571,
                    retention=1.0014090682074324,
                ),
                Run(
                    constraints=Constraints("commander", frozenset()),
                    pool_size=0,
                    base_rate=0.004,
                    precision=None,
                    lift=None,
                    retention=None,
                ),
            ),
        ),
    ),
)


def _runs() -> list[dict[str, Any]]:
    return REPORT.as_dict()["cases"][0]["runs"]


def test_the_report_carries_the_provenance_stamp() -> None:
    stamp = REPORT.as_dict()["provenance"]
    assert stamp["model_id"] == "test/model"
    assert stamp["dim"] == 2
    assert stamp["corpus_updated_at"] == "2026-07-22T21:12:36.682+00:00"
    assert stamp["k"] == 10
    assert stamp["channels"] == list(CHANNELS)


def test_the_report_carries_no_aggregate_across_cases() -> None:
    """A mechanic lift and a theme lift are not commensurable ([ADR 0020])."""
    assert set(REPORT.as_dict()) == {"provenance", "cases"}


def test_colorless_survives_the_json_round_trip() -> None:
    """`None` is unconstrained and `""` is colorless — the JSON must not merge them."""
    assert [run["colors"] for run in _runs()] == [None, "W", ""]


def test_an_undefined_metric_serializes_as_null_never_zero() -> None:
    """The contract the whole report rests on, checked where a reader meets it.

    `metrics.py`'s tests pin this at the arithmetic level. This is the one that
    checks it survives into the file people compare across runs — where a `0`
    would read as a measurement that was taken and came back empty.
    """
    unsatisfiable = _runs()[2]
    assert unsatisfiable["precision"] is None
    assert unsatisfiable["lift"] is None
    assert unsatisfiable["retention"] is None
    assert unsatisfiable["base_rate"] == 0.004, "a real zero-ish number stays a number"


def test_metrics_are_not_rounded_on_the_way_out() -> None:
    """Rounding here would manufacture agreement between runs that differ."""
    assert _runs()[1]["lift"] == 5.167543160690571
    assert _runs()[1]["retention"] == 1.0014090682074324
