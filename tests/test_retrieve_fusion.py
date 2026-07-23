"""Tests for Reciprocal Rank Fusion.

Pure arithmetic over ordinal positions — no store, no model, no frame. The
scores are small enough to hand-compute, which is the point: [ADR 0008] chose
RRF because rank is comparable across channels in a way raw cosine is not, so
the fused score must be checkable without reference to any distance.
"""

from __future__ import annotations

import pytest

from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.config import RRF_K
from mtg_rag.retrieve.fusion import rrf
from mtg_rag.retrieve.search import RankingKey

THEME = PlannedQuery(query_text="graveyard recursion", purpose="theme payoff")
RAMP = PlannedQuery(query_text="mana rocks", purpose="ramp")


def _key(query: PlannedQuery = THEME, channel: str = "oracle") -> RankingKey:
    return RankingKey(query=query, channel=channel)  # type: ignore[arg-type]


def test_rrf_scores_match_the_hand_computed_value() -> None:
    # `alpha` is rank 0 in one ranking and rank 2 in another.
    pool = rrf(
        {
            _key(channel="oracle"): ("alpha", "beta"),
            _key(channel="type"): ("beta", "gamma", "alpha"),
        }
    )

    scores = {candidate.oracle_id: candidate.score for candidate in pool}
    assert scores["alpha"] == pytest.approx(1 / (RRF_K + 0) + 1 / (RRF_K + 2))
    assert scores["beta"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 0))
    assert scores["gamma"] == pytest.approx(1 / (RRF_K + 1))


def test_agreement_across_rankings_outranks_a_single_ranking() -> None:
    # Pins the arithmetic: at equal rank, two contributions beat one. This is
    # not a claim that channel coverage tracks relevance — see issue #44.
    pool = rrf(
        {
            _key(channel="oracle"): ("both", "only_oracle"),
            _key(channel="type"): ("both",),
        }
    )

    assert [candidate.oracle_id for candidate in pool][0] == "both"


def test_pool_is_deduplicated_by_oracle_id() -> None:
    pool = rrf(
        {
            _key(THEME, "oracle"): ("alpha",),
            _key(RAMP, "oracle"): ("alpha",),
            _key(THEME, "type"): ("alpha",),
        }
    )

    assert [candidate.oracle_id for candidate in pool] == ["alpha"]
    assert len(pool[0].sources) == 3


def test_sources_record_query_channel_and_rank() -> None:
    pool = rrf({_key(THEME, "flavor"): ("alpha", "beta")})

    beta = next(candidate for candidate in pool if candidate.oracle_id == "beta")
    (source,) = beta.sources
    assert source.purpose == "theme payoff"
    assert source.channel == "flavor"
    assert source.rank == 1


def test_sources_are_ordered_by_rank_then_channel() -> None:
    # Deterministic provenance, so `--explain` output does not shuffle between
    # runs over identical input.
    pool = rrf(
        {
            _key(THEME, "type"): ("alpha",),
            _key(THEME, "oracle"): ("beta", "alpha"),
            _key(THEME, "flavor"): ("alpha",),
        }
    )

    alpha = next(candidate for candidate in pool if candidate.oracle_id == "alpha")
    assert [(s.channel, s.rank) for s in alpha.sources] == [
        ("flavor", 0),
        ("type", 0),
        ("oracle", 1),
    ]


def test_ties_break_deterministically_on_oracle_id() -> None:
    forwards = rrf({_key(): ("zeta", "alpha")})
    # Same ranks, different insertion order: both cards sit at rank 0 of their
    # own ranking, so only the id can separate them.
    tied = rrf({_key(channel="oracle"): ("zeta",), _key(channel="type"): ("alpha",)})

    assert [c.oracle_id for c in forwards] == ["zeta", "alpha"]  # rank wins
    assert [c.oracle_id for c in tied] == ["alpha", "zeta"]  # id breaks the tie


def test_higher_score_sorts_first() -> None:
    pool = rrf(
        {
            _key(channel="oracle"): ("low", "high"),
            _key(channel="type"): ("high",),
        }
    )
    assert [c.score for c in pool] == sorted((c.score for c in pool), reverse=True)


def test_empty_rankings_fuse_to_an_empty_pool() -> None:
    assert rrf({}) == []


def test_a_channel_that_returned_nothing_contributes_nothing() -> None:
    pool = rrf({_key(channel="oracle"): ("alpha",), _key(channel="flavor"): ()})

    assert [c.oracle_id for c in pool] == ["alpha"]
    assert len(pool[0].sources) == 1


def test_k_is_configurable_and_flattens_rank_differences() -> None:
    # A larger k compresses the gap between ranks; the ADR flags 60 as an
    # inherited default rather than a tuned one, so the knob is exposed.
    ranking = {_key(): ("first", "second")}
    tight = rrf(ranking, k=1)
    loose = rrf(ranking, k=10_000)

    tight_gap = tight[0].score - tight[1].score
    loose_gap = loose[0].score - loose[1].score
    assert loose_gap < tight_gap
