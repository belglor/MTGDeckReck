"""Tests for the per-channel fan-out.

An `EphemeralClient` with hand-written unit-norm 2-D vectors, and a fake encoder
— no model, no network. The vectors point in obvious directions so rank order is
checkable by inspection.

The headline test here is `test_allowlist_ids_missing_from_a_channel_do_not_raise`:
Chroma raises when `ids=` names something a collection does not hold, and the
allowlist comes from the parquet, which covers cards a channel has no text for
([ADR 0014]). Without the intersection, every real request crashes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import chromadb
import numpy as np
import polars as pl
import pytest
from chromadb.api import ClientAPI
from chromadb.config import Settings
from numpy.typing import NDArray

from mtg_rag.embed.config import DOCUMENT_BATCH_SIZE, QUERY_BATCH_SIZE
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.search import channel_allow_ids, search_channels
from mtg_rag.store.chroma import reset_collection, write_vectors
from mtg_rag.store.config import ANONYMIZED_TELEMETRY

EAST = [1.0, 0.0]
NORTH = [0.0, 1.0]


class CountingEncoder:
    """A deterministic encoder that records how it was called.

    The call count is the point: queries must be encoded in one batched call,
    not once per (query, channel) pair.
    """

    def __init__(self) -> None:
        self.dim = 2
        self.calls: list[tuple[str, ...]] = []

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
    ) -> NDArray[np.float32]:  # pragma: no cover - fan-out only encodes queries
        raise AssertionError("retrieval must not encode documents")

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int = QUERY_BATCH_SIZE
    ) -> NDArray[np.float32]:
        self.calls.append(tuple(texts))
        # "east" queries point east, everything else points north.
        return np.array([EAST if "east" in text else NORTH for text in texts], dtype=np.float32)


@pytest.fixture
def client() -> Iterator[ClientAPI]:
    ephemeral = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY, allow_reset=True)
    )
    ephemeral.reset()
    yield ephemeral


def _frame(rows: list[tuple[str, str | None, str | None, str]]) -> pl.DataFrame:
    """A minimal corpus frame: (oracle_id, oracle_text, flavor_text, type_line)."""
    return pl.DataFrame(
        {
            "oracle_id": [r[0] for r in rows],
            "name": [r[0].title() for r in rows],
            "oracle_text": [r[1] for r in rows],
            "flavor_text": [r[2] for r in rows],
            "type_line": [r[3] for r in rows],
            "layout": ["normal"] * len(rows),
            "set_type": ["core"] * len(rows),
        },
        schema_overrides={"oracle_text": pl.String, "flavor_text": pl.String},
    )


# `beta` has no flavor text, so it is absent from the flavor collection — the
# exact shape that makes a parquet-derived allowlist unsafe.
CORPUS = _frame(
    [
        ("alpha", "draw a card", "A memory.", "Instant"),
        ("beta", "gain life", None, "Sorcery"),
        ("gamma", "destroy target", "The end.", "Enchantment"),
    ]
)


def _populate(client: ClientAPI, channel: str, ids: dict[str, list[float]]) -> None:
    collection = reset_collection(client, channel)  # type: ignore[arg-type]
    write_vectors(
        client,
        collection,
        {card_id: np.array(vec, dtype=np.float32) for card_id, vec in ids.items()},
    )


# --- the trap ---------------------------------------------------------------


def test_allowlist_ids_missing_from_a_channel_do_not_raise(client: ClientAPI) -> None:
    # The flavor collection holds only the two cards that have flavor text, but
    # the allowlist covers all three. Passing it unintersected makes Chroma
    # raise `Error finding id`; intersecting first is the whole fix.
    _populate(client, "flavor", {"alpha": EAST, "gamma": NORTH})
    allowed = ["alpha", "beta", "gamma"]

    rankings = search_channels(
        CountingEncoder(),
        client,
        [PlannedQuery(query_text="east things", purpose="theme")],
        frame=CORPUS,
        allow_ids=allowed,
        channels=("flavor",),
        top_k=10,
    )

    # ids in rank order; distances ride along and are not asserted on here.
    assert [tuple(hits) for hits in rankings.values()] == [("alpha", "gamma")]


def test_channel_allow_ids_matches_the_channels_own_ids() -> None:
    # `beta` has no flavor text, so it must not survive into the flavor
    # allowlist even though the constraint filter allowed it.
    allowed = ["alpha", "beta", "gamma"]

    assert channel_allow_ids(CORPUS, "flavor", allowed) == ["alpha", "gamma"]
    assert channel_allow_ids(CORPUS, "oracle", allowed) == ["alpha", "beta", "gamma"]


def test_channel_allow_ids_never_widens_the_allowlist() -> None:
    # The constraint filter is authoritative: a card it excluded cannot come
    # back just because the channel has text for it.
    assert channel_allow_ids(CORPUS, "oracle", ["alpha"]) == ["alpha"]


def test_channel_allow_ids_is_sorted_and_unique() -> None:
    ids = channel_allow_ids(CORPUS, "type", ["gamma", "alpha", "beta", "alpha"])
    assert ids == sorted(set(ids))


# --- fan-out ----------------------------------------------------------------


def test_search_is_constrained_to_the_allowlist(client: ClientAPI) -> None:
    _populate(client, "oracle", {"alpha": EAST, "beta": EAST, "gamma": NORTH})

    rankings = search_channels(
        CountingEncoder(),
        client,
        [PlannedQuery(query_text="east things", purpose="theme")],
        frame=CORPUS,
        allow_ids=["alpha"],
        channels=("oracle",),
        top_k=10,
    )

    assert [tuple(hits) for hits in rankings.values()] == [("alpha",)]


def test_queries_are_encoded_in_one_call(client: ClientAPI) -> None:
    _populate(client, "oracle", {"alpha": EAST, "beta": NORTH})
    _populate(client, "type", {"alpha": EAST, "beta": NORTH})
    encoder = CountingEncoder()

    search_channels(
        encoder,
        client,
        [
            PlannedQuery(query_text="east things", purpose="a"),
            PlannedQuery(query_text="north things", purpose="b"),
        ],
        frame=CORPUS,
        allow_ids=["alpha", "beta"],
        channels=("oracle", "type"),
        top_k=10,
    )

    # Two queries over two channels is one encode call, not four.
    assert encoder.calls == [("east things", "north things")]


def test_each_channel_returns_its_own_ranking(client: ClientAPI) -> None:
    # Same query, deliberately opposite geometry per channel.
    _populate(client, "oracle", {"alpha": EAST, "beta": NORTH})
    _populate(client, "type", {"alpha": NORTH, "beta": EAST})

    rankings = search_channels(
        CountingEncoder(),
        client,
        [PlannedQuery(query_text="east things", purpose="theme")],
        frame=CORPUS,
        allow_ids=["alpha", "beta"],
        channels=("oracle", "type"),
        top_k=10,
    )

    by_channel = {key.channel: tuple(hits) for key, hits in rankings.items()}
    assert by_channel["oracle"][0] == "alpha"
    assert by_channel["type"][0] == "beta"


def test_ranking_keys_carry_query_and_channel(client: ClientAPI) -> None:
    _populate(client, "oracle", {"alpha": EAST})
    query = PlannedQuery(query_text="east things", purpose="theme payoff")

    rankings = search_channels(
        CountingEncoder(),
        client,
        [query],
        frame=CORPUS,
        allow_ids=["alpha"],
        channels=("oracle",),
        top_k=10,
    )

    (key,) = rankings
    assert key.channel == "oracle"
    assert key.query is query  # provenance, not a copy


def test_top_k_bounds_each_ranking(client: ClientAPI) -> None:
    _populate(client, "oracle", {"alpha": EAST, "beta": EAST, "gamma": EAST})

    rankings = search_channels(
        CountingEncoder(),
        client,
        [PlannedQuery(query_text="east things", purpose="theme")],
        frame=CORPUS,
        allow_ids=["alpha", "beta", "gamma"],
        channels=("oracle",),
        top_k=2,
    )

    assert all(len(hits) == 2 for hits in rankings.values())


def test_empty_allowlist_short_circuits_without_querying(client: ClientAPI) -> None:
    _populate(client, "oracle", {"alpha": EAST})

    rankings = search_channels(
        CountingEncoder(),
        client,
        [PlannedQuery(query_text="east things", purpose="theme")],
        frame=CORPUS,
        allow_ids=[],
        channels=("oracle",),
        top_k=10,
    )

    assert all(hits == {} for hits in rankings.values())


def test_no_queries_yields_no_rankings(client: ClientAPI) -> None:
    _populate(client, "oracle", {"alpha": EAST})
    encoder = CountingEncoder()

    rankings = search_channels(
        encoder,
        client,
        [],
        frame=CORPUS,
        allow_ids=["alpha"],
        channels=("oracle",),
        top_k=10,
    )

    assert rankings == {}
    assert encoder.calls == []  # nothing to encode
