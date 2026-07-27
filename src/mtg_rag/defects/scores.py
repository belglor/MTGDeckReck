"""One sweep run's numbers: the checks applied, the results named ([ADR 0026]).

Pure — outputs in, a record out. The stage functions produce a plan and a
recommendation; this scores what they produced, and the runner that drives the
models is a separate concern. That seam is what lets the arithmetic be tested
without a multi-GB download.

Unlike the individual checks, which take their `examples` as an argument to stay
pure, the scorers here wire in the constants the *shipped* prompts actually
show. That is the point of the instrument: it measures the prompt in the repo,
not a prompt handed to it. A rewording that escapes the check is a real change
in what the model is asked, and `test_plan_prompt` / `test_curate_prompt` pin
the constants to the contracts so it cannot happen unnoticed.

Counts sit beside the rates because a rate alone hides its denominator: "0.5
duplicate rate" reads very differently over a two-query plan and a ten-query
one, and #72's plans ranged from 5 to 10 queries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from mtg_rag.curate.prompt import EXAMPLE_RATIONALES
from mtg_rag.curate.recommendation import CuratedCard
from mtg_rag.defects.curate import (
    duplicate_rationale_rate,
    false_type_claim_rate,
    role_count,
    self_quotation_rate,
)
from mtg_rag.defects.curate import parroting_rate as rationale_parroting_rate
from mtg_rag.defects.plan import duplicate_rate, parroting_rate
from mtg_rag.plan.prompt import EXAMPLE_QUERIES


@dataclass(frozen=True, slots=True)
class PlanScores:
    """What one plan scored. `None` rates mean an empty plan measured nothing."""

    query_count: int
    duplicate_rate: float | None
    parroting_rate: float | None


@dataclass(frozen=True, slots=True)
class RecommendationScores:
    """What one recommendation scored.

    The last two are `None` when no returned card could be found in the corpus
    frame, which is different from a run that was checked and came back clean.
    `role_count` is reported, never gated ([ADR 0026], [ADR 0006]).
    """

    card_count: int
    role_count: int | None
    duplicate_rationale_rate: float | None
    parroting_rate: float | None
    self_quotation_rate: float | None
    false_type_claim_rate: float | None


@dataclass(frozen=True, slots=True)
class RunScores:
    """One theme, through both stages, with the theme kept beside its numbers.

    `kind` rides along because #72's finding was a comparison between evocative
    and mechanical phrasing, and a report that loses it cannot show whether that
    gap moved.
    """

    theme: str
    kind: str
    plan: PlanScores
    recommendation: RecommendationScores


def score_plan(queries: Sequence[str]) -> PlanScores:
    """Score one plan's queries against the shipped planner prompt."""
    return PlanScores(
        query_count=len(queries),
        duplicate_rate=duplicate_rate(queries),
        parroting_rate=parroting_rate(queries, examples=EXAMPLE_QUERIES),
    )


def score_recommendation(
    recommendation: Sequence[CuratedCard], rows: pl.DataFrame
) -> RecommendationScores:
    """Score one recommendation against the corpus and the shipped curation prompt.

    `rows` is the hydrated corpus for the returned cards — the same frame
    `curate/render.py` prints from — which the type-claim and self-quotation
    checks need to compare a claim against the card it was made about.
    """
    return RecommendationScores(
        card_count=len(recommendation),
        role_count=role_count(recommendation),
        duplicate_rationale_rate=duplicate_rationale_rate(recommendation),
        parroting_rate=rationale_parroting_rate(recommendation, examples=EXAMPLE_RATIONALES),
        self_quotation_rate=self_quotation_rate(recommendation, rows),
        false_type_claim_rate=false_type_claim_rate(recommendation, rows),
    )
