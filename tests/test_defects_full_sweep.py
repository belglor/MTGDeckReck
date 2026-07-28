"""Tests for driving a theme through the whole pipeline and scoring both stages.

Fixture-driven like `test_retrieve_pool`: the real corpus schema, an ephemeral
Chroma client and a fake encoder, so retrieval is exercised without a model or a
network. The chat model is a canned `LLMClient` answering first with a plan and
then with a recommendation.

What is tested is the orchestration and its failure handling, not the arithmetic
(pinned in `test_defects_scores`) or the checks (pinned in `test_defects_curate`).
The load-bearing behaviour is that **every stage failure is recorded rather than
raised** — an instrument that stops at the first bad output cannot measure how
often output is bad ([ADR 0020]).
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
from mtg_rag.defects.sweep import run_full_sweep
from mtg_rag.embed.channels import channel_frame
from mtg_rag.embed.config import CHANNELS, DOCUMENT_BATCH_SIZE, QUERY_BATCH_SIZE
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.plan.parse import parse_plan
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.retrieve.pool import retrieve
from mtg_rag.store.chroma import reset_collection, write_vectors
from mtg_rag.store.config import ANONYMIZED_TELEMETRY

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"
_THEMES: tuple[tuple[str, str], ...] = (("a spooky graveyard deck that mills itself", "thematic"),)


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    lines = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in lines if line.strip()]
    return build_frame([normalize_card(card) for card in cards])


class LengthEncoder:
    """Encodes to a fixed 2-D query vector — deterministic, no model.

    Mirrors `test_retrieve_pool`'s fake, including its refusal to encode
    documents: retrieval encodes queries only, and a call the other way is a bug
    worth failing loudly on.
    """

    def __init__(self) -> None:
        self.dim = 2

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
    ) -> NDArray[np.float32]:  # pragma: no cover - the pool only encodes queries
        raise AssertionError("retrieval must not encode documents")

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int = QUERY_BATCH_SIZE
    ) -> NDArray[np.float32]:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.fixture
def store(corpus: pl.DataFrame) -> Iterator[ClientAPI]:
    """An index built from the fixture corpus, one collection per channel."""
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


class _FakeClient:
    """Canned `LLMClient`: a plan, then a recommendation, then whatever is queued."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self._responses.pop(0)


def _plan_reply(*queries: str) -> str:
    return json.dumps([{"query_text": q, "purpose": "theme payoff"} for q in queries])


def _recommendation_reply(ids: Sequence[str], *, rationale: str = "Fits the theme well.") -> str:
    return json.dumps(
        [{"oracle_id": i, "role": "theme payoff", "rationale": rationale} for i in ids]
    )


def _run(client: _FakeClient, corpus: pl.DataFrame, store: ClientAPI) -> Any:
    return run_full_sweep(
        format_name="commander",
        client=client,  # type: ignore[arg-type]
        frame=corpus,
        store=store,
        encoder=LengthEncoder(),
        constraints=Constraints(format_name="commander"),
        themes=_THEMES,
        runs=1,
    )


def test_a_completed_run_scores_both_stages(corpus: pl.DataFrame, store: ClientAPI) -> None:
    # The recommendation must name cards from the pool, so the plan is run first
    # to learn what came back — the closed-vocabulary rule ([ADR 0024]) means an
    # invented id would fail validation rather than reach scoring.
    probe = _FakeClient([_plan_reply("graveyard"), _recommendation_reply([])])
    pool_ids = _pool_ids(probe, corpus, store)

    client = _FakeClient([_plan_reply("graveyard"), _recommendation_reply(pool_ids[:2])])
    results = _run(client, corpus, store)

    assert len(results) == 1
    assert results[0].plan is not None
    assert results[0].recommendation is not None
    assert results[0].recommendation.card_count == 2
    assert results[0].error is None


def test_a_plan_that_never_validates_scores_neither_stage(
    corpus: pl.DataFrame, store: ClientAPI
) -> None:
    client = _FakeClient(["not json", "still not json"])

    results = _run(client, corpus, store)

    assert results[0].plan is None
    assert results[0].recommendation is None
    assert results[0].error


def test_a_recommendation_that_never_validates_keeps_the_plan_score(
    corpus: pl.DataFrame, store: ClientAPI
) -> None:
    # The stages fail independently: losing the plan number because curation
    # stumbled would throw away a measurement that was taken successfully.
    client = _FakeClient([_plan_reply("graveyard", "graveyard"), "not json", "still not json"])

    results = _run(client, corpus, store)

    assert results[0].plan is not None
    assert results[0].plan.duplicate_rate == 0.5
    assert results[0].recommendation is None
    assert results[0].error


def test_an_empty_recommendation_is_scored_rather_than_treated_as_a_failure(
    corpus: pl.DataFrame, store: ClientAPI
) -> None:
    # Curation selecting nothing is a valid answer ([ADR 0024]).
    client = _FakeClient([_plan_reply("graveyard"), _recommendation_reply([])])

    results = _run(client, corpus, store)

    assert results[0].recommendation is not None
    assert results[0].recommendation.card_count == 0
    assert results[0].error is None


def test_each_repeat_runs_the_whole_pipeline(corpus: pl.DataFrame, store: ClientAPI) -> None:
    client = _FakeClient(
        [_plan_reply("graveyard"), _recommendation_reply([])] * 3,
    )

    results = run_full_sweep(
        format_name="commander",
        client=client,  # type: ignore[arg-type]
        frame=corpus,
        store=store,
        encoder=LengthEncoder(),
        constraints=Constraints(format_name="commander"),
        themes=_THEMES,
        runs=3,
    )

    assert len(results) == 3
    assert client.calls == 6


def _pool_ids(client: _FakeClient, corpus: pl.DataFrame, store: ClientAPI) -> list[str]:
    """The ids a run actually retrieves, so a test can name real ones.

    Curation's vocabulary is closed to the pool ([ADR 0024]), so a canned
    recommendation naming invented ids would fail validation and this would
    test the parser rather than the sweep.
    """
    queries = parse_plan(_plan_reply("graveyard"))
    pool = retrieve(
        queries,
        constraints=Constraints(format_name="commander"),
        frame=corpus,
        client=store,
        encoder=LengthEncoder(),
    )
    return [candidate.oracle_id for candidate in pool]
