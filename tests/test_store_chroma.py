"""Tests for the Chroma-backed vector store.

An `EphemeralClient` with hand-written vectors — no model, no network, no
persistence. The vectors are unit-norm and two-dimensional so cosine distances
are obvious by inspection: EAST and NEAR_EAST point almost the same way, NORTH
is orthogonal to both.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.config import Settings
from numpy.typing import NDArray

from mtg_rag.store.chroma import (
    open_collection,
    reset_collection,
    search,
    write_vectors,
)
from mtg_rag.store.config import ANONYMIZED_TELEMETRY

EAST = [1.0, 0.0]
NEAR_EAST = [0.92, 0.39]
NORTH = [0.0, 1.0]


def _vectors(*rows: list[float]) -> NDArray[np.float32]:
    return np.array(rows, dtype=np.float32)


def _query(row: list[float]) -> NDArray[np.float32]:
    return np.array(row, dtype=np.float32)


@pytest.fixture
def client() -> Iterator[ClientAPI]:
    # chromadb caches an in-memory system per settings, so an ephemeral client
    # is shared across tests in a process. Reset it to keep tests isolated.
    ephemeral = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY, allow_reset=True)
    )
    ephemeral.reset()
    yield ephemeral


@pytest.fixture
def populated(client: ClientAPI) -> Collection:
    collection = reset_collection(client, "oracle")
    write_vectors(
        client,
        collection,
        ["east", "north", "near_east"],
        _vectors(EAST, NORTH, NEAR_EAST),
    )
    return collection


# --- collection configuration ----------------------------------------------


def test_collection_is_created_without_an_embedding_function(client: ClientAPI) -> None:
    # Chroma's default is DefaultEmbeddingFunction(), which lazily downloads an
    # ONNX MiniLM the first time it is touched. We supply our own vectors, so it
    # must never be attached — on create or on reopen.
    created = reset_collection(client, "oracle")
    assert created.configuration["embedding_function"] is None

    reopened = open_collection(client, "oracle")
    assert reopened.configuration["embedding_function"] is None


def test_collection_uses_cosine_space(client: ClientAPI) -> None:
    # Chroma's default space is l2; the encoder emits unit-norm vectors and
    # ranking is cosine, so this has to be set explicitly.
    collection = reset_collection(client, "oracle")
    hnsw = collection.configuration["hnsw"]

    assert hnsw is not None
    assert hnsw.get("space") == "cosine"


# --- search -----------------------------------------------------------------


def test_search_returns_nearest_id_first(populated: Collection) -> None:
    hits = search(populated, _query(EAST), allow_ids=None, n_results=3)

    assert [card_id for card_id, _ in hits] == ["east", "near_east", "north"]
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)


def test_search_is_constrained_to_the_id_allowlist(populated: Collection) -> None:
    # ADR 0010: filters are evaluated against the parquet and handed here as an
    # allowlist. "east" is the nearest vector, so excluding it proves the
    # allowlist constrains the search rather than filtering afterwards.
    hits = search(populated, _query(EAST), allow_ids=["north"], n_results=3)

    assert [card_id for card_id, _ in hits] == ["north"]


def test_search_with_an_empty_allowlist_returns_nothing(populated: Collection) -> None:
    assert search(populated, _query(EAST), allow_ids=[], n_results=3) == []


def test_search_requests_no_documents_or_metadata(
    populated: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store holds vectors and ids and nothing else (ADR 0010), so asking for
    # documents or metadata would be asking for something that cannot be there.
    captured: dict[str, Any] = {}
    original = populated.query

    def spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(populated, "query", spy)
    search(populated, _query(EAST), allow_ids=None, n_results=1)

    assert "documents" not in captured["include"]
    assert "metadatas" not in captured["include"]


# --- writing ----------------------------------------------------------------


def test_write_chunks_batches_larger_than_the_client_limit(
    client: ClientAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The limit is SQLite-derived and platform-dependent, so it is queried from
    # the client rather than hardcoded. Shrink it and count the add() calls —
    # asserting only on the final count would pass without any chunking.
    monkeypatch.setattr(client, "get_max_batch_size", lambda: 2)
    collection = reset_collection(client, "oracle")
    batch_sizes: list[int] = []
    original_add = collection.add

    def spy_add(**kwargs: Any) -> None:
        batch_sizes.append(len(kwargs["ids"]))
        original_add(**kwargs)

    monkeypatch.setattr(collection, "add", spy_add)

    written = write_vectors(
        client,
        collection,
        ["a", "b", "c", "d", "e"],
        _vectors(EAST, NORTH, NEAR_EAST, EAST, NORTH),
    )

    assert batch_sizes == [2, 2, 1]
    assert written == 5
    assert collection.count() == 5


def test_write_rejects_mismatched_ids_and_vectors(client: ClientAPI) -> None:
    collection = reset_collection(client, "oracle")
    with pytest.raises(ValueError, match="ids"):
        write_vectors(client, collection, ["a", "b"], _vectors(EAST))


# --- reset ------------------------------------------------------------------


def test_reset_collection_discards_previous_vectors(
    client: ClientAPI, populated: Collection
) -> None:
    assert populated.count() == 3
    assert reset_collection(client, "oracle").count() == 0


def test_reset_collection_succeeds_when_none_exists(client: ClientAPI) -> None:
    # Chroma raises NotFoundError when deleting a collection that isn't there,
    # so a first-ever run must not trip over its own cleanup.
    assert reset_collection(client, "flavor").count() == 0
