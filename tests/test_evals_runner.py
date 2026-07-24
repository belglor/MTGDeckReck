"""Tests for the eval runner: what a run produces, and what it refuses to do.

Offline throughout — the fixture corpus, a deterministic stand-in encoder, and an
in-memory Chroma index. See the `client` fixture for why it is built the way
it is — every detail there is load-bearing against a flake.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import polars as pl
import pytest
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb.errors import InternalError
from numpy.typing import NDArray

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.embed.channels import channel_frame
from mtg_rag.embed.config import CHANNELS, DOCUMENT_BATCH_SIZE, QUERY_BATCH_SIZE, Channel
from mtg_rag.embed.index import VectorIndex
from mtg_rag.evals.cases import EvalCase
from mtg_rag.evals.predicates import Predicate
from mtg_rag.evals.runner import CaseResult, Report, Run, run_case, run_cases
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.store.chroma import open_collection, reset_collection, search, write_vectors
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


@pytest.fixture(scope="module")
def client(corpus: pl.DataFrame) -> ClientAPI:
    """An index built from the fixture corpus, once, in memory.

    Three details here are load-bearing against an intermittent Chroma failure
    that made this module fail roughly one run in ten, measured in isolation —
    it was never cross-module, whatever the shared `EphemeralClient` system
    might suggest.

    **In memory, not on disk.** A `PersistentClient` on a private path still
    failed at ~2% with `Nothing found on disk`: the HNSW segment is not on disk
    the instant `write_vectors` returns, and a reader opened before it lands
    finds nothing. An in-memory store cannot miss a file it never writes.

    **Built once per module.** Rebuilding these collections per test only widens
    the window in which that race can be lost. Nothing here mutates the store.

    **`client.reset()` is deliberately not called.** `EphemeralClient` hands
    back one shared system per settings hash, so a reset would wipe collections
    belonging to whichever module runs next.
    """
    store = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY, allow_reset=True)
    )
    for channel in CHANNELS:
        ids = channel_frame(corpus, channel)[ID_COLUMN].to_list()
        collection = reset_collection(store, channel)
        vectors = {
            card_id: np.array([1.0, index / max(len(ids), 1)], dtype=np.float32)
            for index, card_id in enumerate(ids)
        }
        if not vectors:
            continue
        write_vectors(store, collection, vectors)
        assert collection.count() == len(vectors)
        _warm_up(store, channel, sorted(vectors))
    return store


def _warm_up(client: ClientAPI, channel: Channel, ids: list[str], *, attempts: int = 20) -> None:
    """Query through the same handle retrieval uses, until it answers.

    The handle matters. `search_channels` does not reuse the `Collection` that
    wrote the vectors — it opens its own with `open_collection` — and it is that
    fresh handle which intermittently finds nothing just after a write: on a
    persistent store as `Nothing found on disk`, on an ephemeral one as
    `Error finding id`. Warming the fixture's own handle proved nothing about
    it, which is why an earlier version of this still failed at ~6%.

    Only tests hit this shape. `just embed` writes the index and exits; `just
    eval` opens it in a later process, so nothing in production queries a
    collection it wrote moments ago through a second handle.
    """
    probe = np.array([1.0, 0.0], dtype=np.float32)
    for attempt in range(attempts):
        try:
            search(open_collection(client, channel), probe, allow_ids=ids, n_results=1)
        except InternalError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)
        else:
            return


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


def test_run_cases_returns_one_result_per_case(corpus: pl.DataFrame, client: ClientAPI) -> None:
    """The only thing `run_cases` does that `run_case` does not."""
    report = run_cases(
        [_case((Constraints("commander"),)), _case((Constraints("commander"),), keyword="Fuse")],
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
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
