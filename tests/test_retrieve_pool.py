"""Tests for the end-to-end pool: constraints → fan-out → fusion → hydration.

Fixture-driven through `normalize_card` + `build_frame` so the frame is the real
corpus schema, with an ephemeral Chroma client and a fake encoder. No model, no
network.
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
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.retrieve.pool import hydrate, retrieve
from mtg_rag.store.chroma import reset_collection, write_vectors
from mtg_rag.store.config import ANONYMIZED_TELEMETRY

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    lines = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in lines if line.strip()]
    return build_frame([normalize_card(card) for card in cards])


class LengthEncoder:
    """Encodes to a 2-D vector derived from text length — deterministic, no model."""

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
def client(corpus: pl.DataFrame) -> Iterator[ClientAPI]:
    """An index built from the fixture corpus, one collection per channel.

    Vectors are assigned per card so the ranking is deterministic but arbitrary;
    what matters here is which ids can appear, not their order.
    """
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


def _names(corpus: pl.DataFrame, ids: Sequence[str]) -> set[str]:
    return set(corpus.filter(pl.col(ID_COLUMN).is_in(list(ids)))["name"].to_list())


QUERIES = [PlannedQuery(query_text="anything", purpose="theme payoff")]


# --- constraints hold through the whole pipeline ---------------------------


def test_retrieve_returns_only_ids_satisfying_the_constraints(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    pool = retrieve(
        QUERIES,
        constraints=Constraints("commander", platform="paper"),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
    )

    names = _names(corpus, [c.oracle_id for c in pool])
    assert "Black Lotus" not in names  # banned in commander
    assert "Angel of Eternal Dawn" not in names  # arena-only
    assert "Sheep" not in names  # a token, not a real card


def test_color_identity_narrows_the_pool(corpus: pl.DataFrame, client: ClientAPI) -> None:
    pool = retrieve(
        QUERIES,
        constraints=Constraints("commander", frozenset({"G", "W"})),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
    )

    names = _names(corpus, [c.oracle_id for c in pool])
    assert "Sythis, Harvest's Hand" in names  # GW
    assert "Steamflogger Boss" not in names  # red


def test_an_unsatisfiable_request_yields_an_empty_pool(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    # The only commander-legal cards on Arena here are mono-B and mono-G, so a
    # colorless deck has nothing to draw on. An empty pool is a valid answer,
    # not an error — the request is honestly unsatisfiable.
    pool = retrieve(
        QUERIES,
        constraints=Constraints("commander", frozenset(), platform="arena"),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
    )

    assert pool == []


def test_no_queries_yields_an_empty_pool(corpus: pl.DataFrame, client: ClientAPI) -> None:
    pool = retrieve(
        [],
        constraints=Constraints("commander"),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
    )
    assert pool == []


def test_pool_size_caps_the_result(corpus: pl.DataFrame, client: ClientAPI) -> None:
    pool = retrieve(
        QUERIES,
        constraints=Constraints("commander"),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
        pool_size=2,
    )
    assert len(pool) == 2


def test_purpose_survives_from_planned_query_to_candidate(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    pool = retrieve(
        [PlannedQuery(query_text="anything", purpose="a role nobody enumerated")],
        constraints=Constraints("commander"),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
    )

    assert pool
    assert all(
        source.purpose == "a role nobody enumerated"
        for candidate in pool
        for source in candidate.sources
    )


def test_candidates_are_unique_and_ordered_by_score(
    corpus: pl.DataFrame, client: ClientAPI
) -> None:
    pool = retrieve(
        QUERIES,
        constraints=Constraints("commander"),
        frame=corpus,
        client=client,
        encoder=LengthEncoder(),
    )

    ids = [candidate.oracle_id for candidate in pool]
    assert len(ids) == len(set(ids))
    assert [c.score for c in pool] == sorted((c.score for c in pool), reverse=True)


# --- hydration --------------------------------------------------------------


def test_hydrate_preserves_fused_order(corpus: pl.DataFrame) -> None:
    ids = corpus[ID_COLUMN].to_list()
    wanted = [ids[3], ids[0], ids[1]]

    rows = hydrate(corpus, wanted)

    assert rows[ID_COLUMN].to_list() == wanted


def test_hydrate_returns_every_requested_card(corpus: pl.DataFrame) -> None:
    wanted = corpus[ID_COLUMN].to_list()[:5]
    assert hydrate(corpus, wanted).height == len(wanted)


def test_hydrate_carries_the_display_columns(corpus: pl.DataFrame) -> None:
    rows = hydrate(corpus, corpus[ID_COLUMN].to_list()[:1])
    for column in ("name", "mana_cost", "type_line"):
        assert column in rows.columns


def test_hydrate_of_nothing_is_empty(corpus: pl.DataFrame) -> None:
    assert hydrate(corpus, []).height == 0


def test_hydrate_ignores_an_unknown_id(corpus: pl.DataFrame) -> None:
    # Defensive: a card dropped from the corpus since the index was built
    # should not fabricate a row of nulls.
    real = corpus[ID_COLUMN].to_list()[0]
    rows = hydrate(corpus, [real, "not-a-real-oracle-id"])

    assert rows[ID_COLUMN].to_list() == [real]
