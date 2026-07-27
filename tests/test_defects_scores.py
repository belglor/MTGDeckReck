"""Tests for scoring one sweep run ([ADR 0026]).

The individual checks are pinned in `test_defects_text`, `test_defects_plan`
and `test_defects_curate`. What is tested here is the composition: that each
score reads the right check, that the shipped prompt constants are the ones
wired in, and that a partly-unscoreable run reports `None` for the parts it
could not measure rather than a zero it did not earn.

No model and no real corpus — outputs in, a record out.
"""

from __future__ import annotations

import polars as pl

from mtg_rag.curate.prompt import EXAMPLE_RATIONALES
from mtg_rag.curate.recommendation import CuratedCard
from mtg_rag.defects.config import SWEEP_THEMES
from mtg_rag.defects.scores import score_plan, score_recommendation
from mtg_rag.plan.prompt import EXAMPLE_QUERIES


def _rows(*records: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "oracle_id": r["oracle_id"],
                "type_line": r.get("type_line", "Creature — Elf"),
                "oracle_text": r.get("oracle_text", ""),
                "flavor_text": r.get("flavor_text", ""),
            }
            for r in records
        ]
    )


# --- score_plan ------------------------------------------------------------


def test_an_empty_plan_counts_nothing_and_scores_nothing() -> None:
    scores = score_plan([])

    assert scores.query_count == 0
    assert scores.duplicate_rate is None
    assert scores.parroting_rate is None


def test_a_clean_plan_scores_zero_on_both_rates() -> None:
    scores = score_plan(["graveyard recursion", "self-mill enablers"])

    assert scores.query_count == 2
    assert scores.duplicate_rate == 0.0
    assert scores.parroting_rate == 0.0


def test_each_rate_reads_its_own_defect() -> None:
    # One repeat and one copied example, so a scorer wiring the two checks to
    # the same field would show identical numbers and fail here.
    plan = ["mill", "mill", EXAMPLE_QUERIES[0], "graveyard recursion"]
    scores = score_plan(plan)

    assert scores.query_count == 4
    assert scores.duplicate_rate == 0.25
    assert scores.parroting_rate == 0.25


def test_the_shipped_planner_example_is_what_counts_as_parroting() -> None:
    # Proves the scorer measures the prompt in the repo, not one handed to it.
    assert score_plan(list(EXAMPLE_QUERIES)).parroting_rate == 1.0


# --- score_recommendation --------------------------------------------------


def test_an_empty_recommendation_counts_nothing_and_scores_nothing() -> None:
    scores = score_recommendation([], _rows({"oracle_id": "id-1"}))

    assert scores.card_count == 0
    assert scores.role_count is None
    assert scores.duplicate_rationale_rate is None
    assert scores.parroting_rate is None
    assert scores.self_quotation_rate is None
    assert scores.false_type_claim_rate is None


def test_a_clean_recommendation_scores_zero_with_its_roles_counted() -> None:
    cards = [
        CuratedCard(oracle_id="id-1", role="ramp", rationale="Fetches a land to the battlefield."),
        CuratedCard(oracle_id="id-2", role="card draw", rationale="Refills the hand each turn."),
    ]
    rows = _rows(
        {"oracle_id": "id-1", "type_line": "Sorcery"},
        {"oracle_id": "id-2", "type_line": "Enchantment"},
    )

    scores = score_recommendation(cards, rows)

    assert scores.card_count == 2
    assert scores.role_count == 2
    assert scores.duplicate_rationale_rate == 0.0
    assert scores.parroting_rate == 0.0
    assert scores.self_quotation_rate == 0.0
    assert scores.false_type_claim_rate == 0.0


def test_a_run_exhibiting_every_defect_scores_each_one_separately() -> None:
    # #91's angels run in miniature: one collapsed role, the example parroted
    # onto the top card, a stamped rationale, a quoted flavor line, and a
    # non-creature called a creature.
    cards = [
        CuratedCard(oracle_id="id-1", role="theme payoff", rationale=EXAMPLE_RATIONALES[0]),
        CuratedCard(oracle_id="id-2", role="theme payoff", rationale="It is a radiant creature."),
        CuratedCard(oracle_id="id-3", role="theme payoff", rationale="It is a radiant creature."),
        CuratedCard(
            oracle_id="id-4", role="theme payoff", rationale="An apocalypse in dragon form."
        ),
    ]
    rows = _rows(
        {"oracle_id": "id-1", "type_line": "Sorcery"},
        {"oracle_id": "id-2", "type_line": "Instant"},
        {"oracle_id": "id-3", "type_line": "Creature — Angel"},
        {"oracle_id": "id-4", "flavor_text": "An apocalypse in dragon form."},
    )

    scores = score_recommendation(cards, rows)

    assert scores.card_count == 4
    assert scores.role_count == 1
    assert scores.duplicate_rationale_rate == 0.25
    assert scores.parroting_rate == 0.25
    assert scores.self_quotation_rate == 0.25
    assert scores.false_type_claim_rate == 0.25


def test_a_recommendation_the_corpus_cannot_check_still_scores_what_it_can() -> None:
    # The two corpus-dependent checks go undefined; the three that read only the
    # recommendation still report. A scorer that short-circuited on the frame
    # would lose them.
    cards = [
        CuratedCard(oracle_id="gone-1", role="theme payoff", rationale="Same."),
        CuratedCard(oracle_id="gone-2", role="theme payoff", rationale="Same."),
    ]

    scores = score_recommendation(cards, _rows({"oracle_id": "id-1"}))

    assert scores.card_count == 2
    assert scores.role_count == 1
    assert scores.duplicate_rationale_rate == 0.5
    assert scores.parroting_rate == 0.0
    assert scores.self_quotation_rate is None
    assert scores.false_type_claim_rate is None


def test_the_shipped_curation_example_is_what_counts_as_parroting() -> None:
    parroted = [CuratedCard(oracle_id="id-1", role="theme payoff", rationale=EXAMPLE_RATIONALES[0])]

    scores = score_recommendation(parroted, _rows({"oracle_id": "id-1"}))

    assert scores.parroting_rate == 1.0


# --- the theme set ---------------------------------------------------------


def test_the_sweep_themes_are_the_ten_recorded_prompts() -> None:
    # Changing this set starts a new baseline, so the count and the evocative /
    # mechanical split are pinned rather than left to drift silently.
    assert len(SWEEP_THEMES) == 10
    assert sum(kind == "thematic" for _, kind in SWEEP_THEMES) == 5
    assert sum(kind == "mechanical" for _, kind in SWEEP_THEMES) == 5
    assert len({theme for theme, _ in SWEEP_THEMES}) == 10
