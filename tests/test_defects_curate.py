"""Tests for the recommendation defect checks ([ADR 0026]).

Every case is a hand-built `CuratedCard` list plus a small corpus frame — no
model, no real corpus. The three checks that compare a claim against a card
take the hydrated rows the same way `curate/render.py` does.

What they are for: #91 records the prompt's worked example returned as the top
card's rationale in 5 of 5 runs, one rationale stamped across 20+ of 30 cards,
flavor text handed back as an argument, `Illumination` (an Instant) called "a
powerful and radiant creature", and role grouping collapsing to one bucket.
Each is a matter of fact rather than taste, which is what makes it measurable
([ADR 0026]) — nothing here judges whether a rationale is any *good*.
"""

from __future__ import annotations

import polars as pl
import pytest

from mtg_rag.curate.prompt import EXAMPLE_RATIONALES
from mtg_rag.curate.recommendation import CuratedCard
from mtg_rag.defects.curate import (
    duplicate_rationale_rate,
    false_type_claim_rate,
    parroting_rate,
    role_count,
    self_quotation_rate,
)


def _card(
    oracle_id: str = "id-1",
    role: str = "theme payoff",
    rationale: str = "Fits.",
) -> CuratedCard:
    return CuratedCard(oracle_id=oracle_id, role=role, rationale=rationale)


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


# --- role_count ------------------------------------------------------------


def test_an_empty_recommendation_has_no_defined_role_count() -> None:
    assert role_count([]) is None


def test_every_card_under_one_role_counts_one() -> None:
    # #91's collapse: 4 of 5 runs put all 30 cards under `theme payoff`.
    assert role_count([_card(role="theme payoff") for _ in range(30)]) == 1


def test_distinct_roles_are_counted() -> None:
    roles = ["ramp", "card draw", "removal / interaction", "theme payoff"]

    assert role_count([_card(role=role) for role in roles]) == 4


def test_roles_differing_only_in_case_are_one_job() -> None:
    # The question is how many distinct jobs curation named, not how many
    # headings `render.py` would print (it groups on the exact string).
    assert role_count([_card(role="Theme Payoff"), _card(role="theme payoff")]) == 1


# --- duplicate_rationale_rate and parroting_rate ---------------------------


def test_an_empty_recommendation_has_no_defined_duplicate_rationale_rate() -> None:
    assert duplicate_rationale_rate([]) is None


def test_one_rationale_stamped_across_every_card_scores_almost_one() -> None:
    stamped = [
        _card(oracle_id=f"id-{i}", rationale="A powerful and radiant card.") for i in range(4)
    ]

    assert duplicate_rationale_rate(stamped) == 0.75


def test_distinct_rationales_score_no_duplication() -> None:
    cards = [
        _card(oracle_id="id-1", rationale="Recurs lands."),
        _card(oracle_id="id-2", rationale="Mills three."),
    ]

    assert duplicate_rationale_rate(cards) == 0.0


def test_the_shipped_example_rationale_is_what_a_parroting_recommendation_returns() -> None:
    # #91's headline finding, end to end against the real constant: the check
    # is wired to the string the prompt actually shows.
    parroted = [_card(rationale=EXAMPLE_RATIONALES[0])]

    assert parroting_rate(parroted, examples=EXAMPLE_RATIONALES) == 1.0


def test_a_real_rationale_scores_no_parroting() -> None:
    cards = [_card(rationale="Recurs a land from the graveyard every upkeep.")]

    assert parroting_rate(cards, examples=EXAMPLE_RATIONALES) == 0.0


# --- self_quotation_rate ---------------------------------------------------


def test_a_rationale_quoting_the_cards_flavor_text_is_caught() -> None:
    # #91 recorded "An apocalypse in dragon form." — a flavor line — returned
    # as the argument for why a card fits.
    cards = [_card(oracle_id="id-1", rationale="An apocalypse in dragon form.")]
    rows = _rows({"oracle_id": "id-1", "flavor_text": "An apocalypse in dragon form."})

    assert self_quotation_rate(cards, rows) == 1.0


def test_a_rationale_quoting_the_cards_oracle_text_is_caught() -> None:
    cards = [_card(oracle_id="id-1", rationale="Mills you every upkeep.")]
    printed = "At the beginning of your upkeep, mills you every upkeep."
    rows = _rows({"oracle_id": "id-1", "oracle_text": printed})

    assert self_quotation_rate(cards, rows) == 1.0


def test_a_paraphrase_is_not_quotation() -> None:
    # Describing what a card does is the rationale's job ([ADR 0005]); only
    # handing the printed text back is the defect.
    cards = [_card(oracle_id="id-1", rationale="Fills the graveyard the deck feeds on.")]
    printed = "At the beginning of your upkeep, mill three cards."
    rows = _rows({"oracle_id": "id-1", "oracle_text": printed})

    assert self_quotation_rate(cards, rows) == 0.0


def test_a_card_missing_from_the_corpus_is_not_scored() -> None:
    # Mirrors `render.py`'s guard: an id the frame no longer carries is skipped,
    # and the denominator is what could actually be checked.
    cards = [
        _card(oracle_id="id-1", rationale="An apocalypse in dragon form."),
        _card(oracle_id="gone", rationale="Anything."),
    ]
    rows = _rows({"oracle_id": "id-1", "flavor_text": "An apocalypse in dragon form."})

    assert self_quotation_rate(cards, rows) == 1.0


def test_a_recommendation_with_nothing_checkable_is_undefined() -> None:
    # Not a clean zero it did not earn.
    cards = [_card(oracle_id="gone")]

    assert self_quotation_rate(cards, _rows({"oracle_id": "id-1"})) is None


def test_a_card_with_no_printed_text_cannot_be_quoted() -> None:
    # A vanilla creature has no oracle or flavor text; empty must not match
    # every rationale by being a substring of nothing.
    cards = [_card(oracle_id="id-1", rationale="A bear.")]
    rows = _rows({"oracle_id": "id-1", "oracle_text": "", "flavor_text": ""})

    assert self_quotation_rate(cards, rows) == 0.0


# --- false_type_claim_rate -------------------------------------------------


def test_calling_an_instant_a_creature_is_caught() -> None:
    # #91's exact case: `Illumination` is an Instant.
    cards = [_card(oracle_id="id-1", rationale="Illumination is a powerful and radiant creature.")]
    rows = _rows({"oracle_id": "id-1", "type_line": "Instant"})

    assert false_type_claim_rate(cards, rows) == 1.0


def test_calling_a_creature_a_creature_is_not_a_defect() -> None:
    cards = [_card(oracle_id="id-1", rationale="This is a creature that fits the theme.")]
    rows = _rows({"oracle_id": "id-1", "type_line": "Creature — Elf Druid"})

    assert false_type_claim_rate(cards, rows) == 0.0


def test_naming_a_type_the_card_acts_on_is_not_a_claim_about_the_card() -> None:
    # The measured trap: 44.8% of commander-legal cards name a type they are not
    # in their own text, so a substring scan would flag any faithful rationale
    # for nearly half the corpus. Only a copular claim counts.
    cards = [_card(oracle_id="id-1", rationale="Exiles target creature at instant speed.")]
    rows = _rows({"oracle_id": "id-1", "type_line": "Instant"})

    assert false_type_claim_rate(cards, rows) == 0.0


def test_a_subtype_is_not_a_false_claim() -> None:
    # "a powerful Equipment" for an Artifact — Equipment is correct shorthand,
    # which is why only the eight top-level types are in CARD_TYPES.
    cards = [_card(oracle_id="id-1", rationale="It is a powerful Equipment for the deck.")]
    rows = _rows({"oracle_id": "id-1", "type_line": "Artifact — Equipment"})

    assert false_type_claim_rate(cards, rows) == 0.0


def test_a_multi_type_card_is_judged_against_all_its_types() -> None:
    cards = [_card(oracle_id="id-1", rationale="It is an artifact that ramps.")]
    rows = _rows({"oracle_id": "id-1", "type_line": "Artifact Creature — Golem"})

    assert false_type_claim_rate(cards, rows) == 0.0


def test_the_rate_is_over_the_cards_that_could_be_checked() -> None:
    cards = [
        _card(oracle_id="id-1", rationale="It is a creature."),
        _card(oracle_id="id-2", rationale="Draws cards."),
    ]
    rows = _rows(
        {"oracle_id": "id-1", "type_line": "Instant"},
        {"oracle_id": "id-2", "type_line": "Sorcery"},
    )

    assert false_type_claim_rate(cards, rows) == pytest.approx(0.5)
