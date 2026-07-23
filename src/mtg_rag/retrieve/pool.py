"""The whole retrieval path: constraints → allowlist → fan-out → fusion → cards.

This is where the two halves of `retrieve/` meet. The hard constraints produce
the ids a search may return ([ADR 0001]), the fan-out asks each channel, fusion
collapses the rankings into one pool ([ADR 0008]), and hydration reads the cards
back from the parquet — never from the store, which holds vectors and nothing
else ([ADR 0010]).

No LLM is called here. The plan arrives as data and the pool leaves as data;
what produces the one and consumes the other are separate concerns.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from chromadb.api import ClientAPI

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.embed.config import CHANNELS, Channel
from mtg_rag.embed.encoder import Encoder
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.config import CHANNEL_TOP_K, DEFAULT_POOL_SIZE
from mtg_rag.retrieve.filters import Constraints, allowed_ids
from mtg_rag.retrieve.fusion import Candidate, rrf
from mtg_rag.retrieve.search import search_channels


def retrieve(
    queries: Sequence[PlannedQuery],
    *,
    constraints: Constraints,
    frame: pl.DataFrame,
    client: ClientAPI,
    encoder: Encoder,
    channels: Sequence[Channel] = CHANNELS,
    top_k: int = CHANNEL_TOP_K,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> list[Candidate]:
    """The candidate pool for `queries` under `constraints`, best first.

    An empty pool is a valid answer — a request can be legitimately
    unsatisfiable (a mono-white deck asking for black removal), and that is
    honest output rather than an error.
    """
    permitted = allowed_ids(frame, constraints)
    rankings = search_channels(
        encoder,
        client,
        queries,
        frame=frame,
        allow_ids=permitted,
        channels=channels,
        top_k=top_k,
    )
    return rrf(rankings)[:pool_size]


def hydrate(frame: pl.DataFrame, ids: Sequence[str]) -> pl.DataFrame:
    """The corpus rows for `ids`, **in the order given**.

    A join rather than a filter: `filter` returns rows in corpus order and would
    silently discard the ranking, which is the one thing the pool exists to
    produce. Ids the corpus does not hold are dropped rather than filled with
    nulls — a card that left the corpus since the index was built is absent, not
    a row of blanks.
    """
    wanted = pl.DataFrame({ID_COLUMN: list(ids)}, schema={ID_COLUMN: pl.String})
    return wanted.join(frame, on=ID_COLUMN, how="inner", maintain_order="left")
