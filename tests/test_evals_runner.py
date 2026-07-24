"""Tests for the eval runner: what a run produces, and what it refuses to do.

Offline throughout — the fixture corpus, a deterministic stand-in encoder, and
an ephemeral Chroma client, following `tests/test_retrieve_pool.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import polars as pl
import pytest
from chromadb.api import ClientAPI
from chromadb.config import Settings
from numpy.typing import NDArray

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.embed.channels import channel_frame
from mtg_rag.embed.config import CHANNELS, DOCUMENT_BATCH_SIZE, QUERY_BATCH_SIZE
from mtg_rag.embed.index import VectorIndex
from mtg_rag.evals.cases import EvalCase, Predicate
from mtg_rag.evals.runner import Provenance, run_case, run_cases
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.store.chroma import reset_collection, write_vectors
from mtg_rag.store.config import ANONYMIZED_TELEMETRY

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    lines = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in lines if line.strip()]
    return build_frame([normalize_card(card) for card in cards])


class LengthEncoder:
    """Deterministic stand-in — no model, no download."""

    def __init__(self) -> None:
        self.dim = 2

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
    ) -> NDArray[np.float32]:  # pragma: no cover - the eval only encodes queries
        raise AssertionError("the eval must not encode documents")

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int = QUERY_BATCH_SIZE
    ) -> NDArray[np.float32]:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.fixture
def client(corpus: pl.DataFrame) -> Iterator[ClientAPI]:
    ephemeral = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY, allow_reset=True)
    )
    ephemeral.reset()
    for channel in CHANNELS:
        ids = channel_frame(corpus, channel)[ID_COLUMN].to_list()
        collection = reset_collection(ephemeral, channel)
        vectors = {
            card_id: np.array([1.0, index / max(len(ids), 1)], dtype=np.float32)
            for index, card_id in enumerate(ids)
        }
        if vectors:
            write_vectors(ephemeral, collection, vectors)
    yield ephemeral


INDEX = VectorIndex(
    model_id="test/model",
    dim=2,
    corpus_updated_at="2026-07-22T21:12:36.682+00:00",
    corpus_row_count=17,
    channel_counts={"oracle": 17, "flavor": 3, "type": 17},
    embedded_at="2026-07-23T20:40:10.699755+00:00",
)


def _case(constraints: tuple[Constraints, ...], *, keyword: str = "Cycling") -> EvalCase:
    return EvalCase(
        id="cycling",
        query="cycling",
        rationale="Scryfall's keywords column is the ground truth.",
        predicate=Predicate(kind="keyword", value=keyword),
        constraints=constraints,
    )


def _run(case: EvalCase, corpus: pl.DataFrame, client: ClientAPI, k: int = 10):
    return run_case(
        case,
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
        k=k,
        channels=CHANNELS,
    )


# --- what a run produces ---------------------------------------------------


def test_one_run_per_constraint_set_in_file_order(corpus: pl.DataFrame, client: ClientAPI) -> None:
    case = _case(
        (
            Constraints("commander"),
            Constraints("commander", frozenset({"W"})),
        )
    )
    result = _run(case, corpus, client)
    assert [run.constraints.color_identity for run in result.runs] == [None, frozenset({"W"})]


def test_the_first_constraint_set_is_the_retention_reference(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    result = _run(_case((Constraints("commander"),)), corpus, client)
    assert result.runs[0].retention == pytest.approx(1.0)


def test_retention_is_measured_against_the_first_run(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    case = _case((Constraints("commander"), Constraints("commander", frozenset({"W"}))))
    reference, tighter = _run(case, corpus, client).runs
    assert reference.lift is not None and tighter.lift is not None
    assert tighter.retention == pytest.approx(tighter.lift / reference.lift)


def test_base_rate_differs_between_constraint_sets(corpus: pl.DataFrame, client: ClientAPI) -> None:
    """The denominator must follow the constraint, or lift means nothing."""
    case = _case((Constraints("commander"), Constraints("commander", frozenset({"W"}))))
    reference, tighter = _run(case, corpus, client).runs
    assert reference.base_rate != tighter.base_rate


def test_pool_size_is_what_came_back_not_what_was_asked_for(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    """The fixture corpus is far smaller than k, so the pool must be short."""
    result = _run(_case((Constraints("commander"),)), corpus, client, k=500)
    assert 0 < result.runs[0].pool_size < 500


# --- what a run refuses to do ----------------------------------------------


def test_a_poor_metric_does_not_fail_the_run(corpus: pl.DataFrame, client: ClientAPI) -> None:
    """A number the reader dislikes is the output, not an error ([ADR 0011])."""
    case = _case((Constraints("commander"),), keyword="Fuse")
    result = _run(case, corpus, client)
    assert result.runs[0].precision is not None


def test_an_unsatisfiable_constraint_reports_rather_than_raises(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    """Colorless-only against a coloured predicate: an empty pool is honest output."""
    case = _case((Constraints("commander", frozenset()),))
    result = _run(case, corpus, client)
    run = result.runs[0]
    assert run.precision is None or run.precision == 0.0


def test_the_report_carries_the_provenance_stamp(corpus: pl.DataFrame, client: ClientAPI) -> None:
    provenance = Provenance.from_index(INDEX, k=10, channels=CHANNELS)
    report = run_cases(
        [_case((Constraints("commander"),))],
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
        provenance=provenance,
        channels=CHANNELS,
    )
    stamp = report.as_dict()["provenance"]
    assert stamp["model_id"] == "test/model"
    assert stamp["dim"] == 2
    assert stamp["corpus_updated_at"] == "2026-07-22T21:12:36.682+00:00"
    assert stamp["k"] == 10
    assert stamp["channels"] == list(CHANNELS)


def test_the_report_carries_no_aggregate_across_cases(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    """A mechanic lift and a theme lift are not commensurable ([ADR 0020])."""
    report = run_cases(
        [_case((Constraints("commander"),))],
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
        provenance=Provenance.from_index(INDEX, k=10, channels=CHANNELS),
        channels=CHANNELS,
    )
    assert set(report.as_dict()) == {"provenance", "cases"}


def test_colorless_survives_the_json_round_trip(corpus: pl.DataFrame, client: ClientAPI) -> None:
    """`None` is unconstrained and `""` is colorless — the JSON must not merge them."""
    case = _case((Constraints("commander"), Constraints("commander", frozenset())))
    report = run_cases(
        [case],
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
        provenance=Provenance.from_index(INDEX, k=10, channels=CHANNELS),
        channels=CHANNELS,
    )
    colors = [run["colors"] for run in report.as_dict()["cases"][0]["runs"]]
    assert colors == [None, ""]
