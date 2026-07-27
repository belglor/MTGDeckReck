"""Tests for curation's typed output schema.

`CuratedCard` is the boundary [ADR 0024] asks for: malformed curation output
fails here rather than degrading into a subtly wrong recommendation. Nothing
in these tests touches a model or the retrieved pool — the schema is data, and
the LLM call and pool-membership check are later issues.
"""

from __future__ import annotations

import dataclasses

import pytest

from mtg_rag.curate.recommendation import CuratedCard


def test_a_well_formed_card_keeps_all_fields() -> None:
    card = CuratedCard(
        oracle_id="abc123", role="payoff", rationale="recurs lands from the graveyard"
    )

    assert card.oracle_id == "abc123"
    assert card.role == "payoff"
    assert card.rationale == "recurs lands from the graveyard"


def test_curated_card_requires_non_empty_role() -> None:
    with pytest.raises(ValueError, match="role"):
        CuratedCard(oracle_id="abc123", role="", rationale="fits the theme")


def test_curated_card_requires_non_empty_rationale() -> None:
    with pytest.raises(ValueError, match="rationale"):
        CuratedCard(oracle_id="abc123", role="payoff", rationale="")


def test_whitespace_only_fields_are_rejected() -> None:
    # A model emitting "  " satisfies a naive truthiness check but says nothing.
    with pytest.raises(ValueError, match="role"):
        CuratedCard(oracle_id="abc123", role="   ", rationale="fits the theme")
    with pytest.raises(ValueError, match="rationale"):
        CuratedCard(oracle_id="abc123", role="payoff", rationale="\t")


def test_curated_card_is_frozen() -> None:
    card = CuratedCard(oracle_id="abc123", role="payoff", rationale="fits the theme")

    with pytest.raises(dataclasses.FrozenInstanceError):
        card.role = "something else"  # type: ignore[misc]


def test_oracle_id_emptiness_is_not_checked_here() -> None:
    # oracle_id is closed-vocabulary against the retrieved pool, not a
    # non-empty check — that validation needs the pool and belongs to the
    # parser at the boundary ([ADR 0024]), not to construction.
    card = CuratedCard(oracle_id="", role="payoff", rationale="fits the theme")

    assert card.oracle_id == ""
