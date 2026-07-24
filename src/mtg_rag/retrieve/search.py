"""Fan a plan out across the embedding channels, one ranking per (query, channel).

Each ranking is `store.search`'s rank-ordered mapping of `oracle_id` to
distance, passed through unchanged. **Rank is the payload; the distance is
display only.** Fusion reads ordinal position and nothing else, because cosine
scores from different channels index different registers of text over different
subsets of the corpus and are not commensurable ([ADR 0008]) — a distance must
never reach a ranking decision. It is carried because we already paid to fetch
it and it answers a question rank cannot: whether a channel's top hit was a
confident match or the best of a bad set. `--explain` shows it.

**The allowlist must be intersected with each channel's own ids before it is
used.** The constraint filter derives its allowlist from the parquet, which
covers every real card; a channel's collection covers only the cards that have
text there ([ADR 0014]). Chroma raises when `ids=` names something a collection
does not hold, so an unintersected allowlist crashes every real request rather
than quietly over-fetching. `embed.channels.channel_frame` gives the channel's
id set exactly — it is the function that built the collection — for a few
milliseconds, against hundreds for asking the store.

That equality holds only while the index is current with the parquet. If they
drift, this reintroduces the very crash it prevents; `data/vectors.meta.json`
records which corpus the index was built from, and `just embed` restores it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import polars as pl
from chromadb.api import ClientAPI

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.embed.channels import channel_frame
from mtg_rag.embed.config import Channel
from mtg_rag.embed.encoder import Encoder
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.store.chroma import open_collection, search

#: One channel's answer to one query: `oracle_id` to distance, **nearest first**.
#: Iteration order carries the rank, which is the part fusion uses ([ADR 0008]).
type Ranking = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RankingKey:
    """Which query and which channel produced a ranking.

    Fusion collapses the rankings into one pool, so this is what lets a
    candidate still say why it was found — the query's `purpose` is curation's
    starting hypothesis for the card's role ([ADR 0005]).
    """

    query: PlannedQuery
    channel: Channel


def channel_allow_ids(frame: pl.DataFrame, channel: Channel, allowed: Collection[str]) -> list[str]:
    """`allowed`, narrowed to the cards this channel actually holds.

    Sorted for determinism. Narrowing only: a card the constraints excluded
    cannot return just because the channel has text for it.
    """
    in_channel = set(channel_frame(frame, channel)[ID_COLUMN].to_list())
    return sorted(in_channel.intersection(allowed))


def search_channels(
    encoder: Encoder,
    client: ClientAPI,
    queries: Sequence[PlannedQuery],
    *,
    frame: pl.DataFrame,
    allow_ids: Collection[str],
    channels: Sequence[Channel],
    top_k: int,
) -> dict[RankingKey, Ranking]:
    """One ranking per (query, channel): `oracle_id` to distance, nearest first.

    Iteration order is rank order — that is `store.search`'s documented contract
    and what fusion consumes. The distances ride along for display and must not
    be read as a score (see the module docstring).

    Every query text is encoded in a single batched call, then searched channel
    by channel. **The loop is sequential on purpose.** An allowlisted search
    measures 4–7 ms, so a 3-channel × 8-query plan runs in roughly 130 ms; a
    thread pool buys nothing at that cost and adds a failure mode. [ADR 0004]
    says the app executes a plan "in parallel" — that is about the queries being
    independent, which they are, not a requirement to spend threads.
    """
    if not queries:
        return {}

    vectors = encoder.encode_queries([query.query_text for query in queries])

    rankings: dict[RankingKey, Ranking] = {}
    for channel in channels:
        permitted = channel_allow_ids(frame, channel, allow_ids)
        collection = open_collection(client, channel)
        for query, vector in zip(queries, vectors, strict=True):
            hits = search(collection, vector, allow_ids=permitted, n_results=top_k)
            rankings[RankingKey(query=query, channel=channel)] = hits
    return rankings
