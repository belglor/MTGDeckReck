"""Combine many rankings into one candidate pool by Reciprocal Rank Fusion.

Each ranking contributes `1 / (k + rank)` to every card it names, and a card's
score is the sum over the rankings that found it. Only ordinal position is read
([ADR 0008]): cosine scores from different channels index different registers of
text over different subsets of the corpus, so they are not commensurable and
must never be averaged, summed, or maxed.

Channels are weighted uniformly. That is a knowing choice, not an oversight —
[ADR 0008] names the golden set as the instrument for deciding weights, and it
does not exist yet. Issue #44 records the limitation this leaves in place: a
card absent from a channel earns nothing there, and channel coverage is uneven
for reasons unrelated to relevance.

Fusion is also what dedupes. A card found by three queries is one candidate
carrying three sources, which is what lets curation see *why* it was retrieved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mtg_rag.embed.config import Channel
from mtg_rag.retrieve.config import RRF_K
from mtg_rag.retrieve.search import RankingKey


@dataclass(frozen=True, slots=True)
class Source:
    """One ranking that found a card, and where in it the card sat."""

    purpose: str
    channel: Channel
    rank: int


@dataclass(frozen=True, slots=True)
class Candidate:
    """A card in the pool, with its fused score and everything that found it."""

    oracle_id: str
    score: float
    sources: tuple[Source, ...]


def rrf(rankings: Mapping[RankingKey, Sequence[str]], *, k: int = RRF_K) -> list[Candidate]:
    """Fuse `rankings` into one pool, best first.

    Ordering is by score descending, then `oracle_id` — two cards with identical
    scores would otherwise come out in dictionary order, making the pool depend
    on the order the rankings happened to be built in.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[Source]] = {}

    for key, ids in rankings.items():
        for rank, oracle_id in enumerate(ids):
            scores[oracle_id] = scores.get(oracle_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(oracle_id, []).append(
                Source(purpose=key.query.purpose, channel=key.channel, rank=rank)
            )

    return [
        Candidate(
            oracle_id=oracle_id,
            score=score,
            # Sorted so `--explain` reads the same way across runs.
            sources=tuple(sorted(sources[oracle_id], key=lambda s: (s.rank, s.channel))),
        )
        for oracle_id, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
