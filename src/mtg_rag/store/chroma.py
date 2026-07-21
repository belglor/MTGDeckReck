"""The Chroma-backed vector store: vectors keyed by `oracle_id`, and nothing
else.

No card metadata lives here — not legality, not colour identity, not card text
([ADR 0010]). Filters are evaluated against the corpus parquet, which is the
single source of truth, and the surviving ids are handed to `search` as an
allowlist. That is why a legality or price change costs nothing here: rewriting
the parquet is the whole job.

One collection per channel, each rebuilt wholesale rather than reconciled
([ADR 0015]). A card absent from a channel simply has no id in that collection,
which is normal rather than an error ([ADR 0014]).
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from numpy.typing import NDArray

from mtg_rag.embed.config import Channel
from mtg_rag.store.config import ANONYMIZED_TELEMETRY, COLLECTION_PREFIX, DISTANCE_SPACE


def collection_name(channel: Channel) -> str:
    """`"oracle"` -> `"cards_oracle"`."""
    return f"{COLLECTION_PREFIX}{channel}"


def open_client(vector_dir: Path) -> ClientAPI:
    """A persistent client rooted at `vector_dir`, with telemetry off."""
    vector_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(vector_dir),
        settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY),
    )


def _create(client: ClientAPI, channel: Channel) -> Collection:
    """Create a channel's collection.

    `embedding_function=None` is load-bearing: the default is
    `DefaultEmbeddingFunction()`, which lazily downloads an ONNX MiniLM the
    first time it is touched. We always supply our own vectors, so nothing
    should ever be embedded here.
    """
    return client.create_collection(
        collection_name(channel),
        embedding_function=None,
        configuration={"hnsw": {"space": DISTANCE_SPACE}},
    )


def reset_collection(client: ClientAPI, channel: Channel) -> Collection:
    """Drop the channel's collection and return a fresh empty one.

    Rebuilding wholesale is what keeps the index from drifting out of step with
    the parquet ([ADR 0015]) — there is no incremental state to get wrong.
    Deleting a collection that was never created raises, which a first-ever run
    would otherwise trip over.
    """
    with suppress(NotFoundError):
        client.delete_collection(collection_name(channel))
    return _create(client, channel)


def open_collection(client: ClientAPI, channel: Channel) -> Collection:
    """Open an existing channel collection for reading.

    `embedding_function=None` matters on the read path too, for the same reason
    it matters on create.
    """
    return client.get_collection(collection_name(channel), embedding_function=None)


def write_vectors(
    client: ClientAPI,
    collection: Collection,
    ids: Sequence[str],
    vectors: NDArray[np.float32],
) -> int:
    """Add `vectors` under `ids`, in chunks the client will accept.

    The maximum batch size is SQLite-derived and platform-dependent, so it is
    queried from the client rather than hardcoded.
    """
    if len(ids) != len(vectors):
        raise ValueError(f"{len(ids)} ids but {len(vectors)} vectors — they must correspond")

    limit = client.get_max_batch_size()
    for start in range(0, len(ids), limit):
        chunk = slice(start, start + limit)
        collection.add(ids=list(ids[chunk]), embeddings=vectors[chunk])
    return len(ids)


def search(
    collection: Collection,
    query_vector: NDArray[np.float32],
    *,
    allow_ids: Sequence[str] | None,
    n_results: int,
) -> list[tuple[str, float]]:
    """Nearest ids to `query_vector`, closest first, as `(oracle_id, distance)`.

    `allow_ids` constrains the search before ranking ([ADR 0010]); `None` means
    unconstrained. Chroma pre-filters, so a constrained query returns a full
    `n_results` from within the allowlist rather than filtering a top-k
    afterwards and coming up short.

    Only distances are requested — the collection holds no documents or
    metadata to return, and asking for them would be asking for something that
    cannot be there.
    """
    if allow_ids is not None and not allow_ids:
        # Nothing is permitted, so there is nothing to ask. Chroma handles an
        # empty allowlist correctly; this just skips a pointless round trip.
        return []

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        ids=list(allow_ids) if allow_ids is not None else None,
        include=["distances"],
    )

    distances = result["distances"]
    if distances is None:  # pragma: no cover - we always ask for distances
        raise RuntimeError("chroma returned no distances despite include=['distances']")
    return list(zip(result["ids"][0], distances[0], strict=True))
