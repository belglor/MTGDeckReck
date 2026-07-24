"""Tests for the planner's typed query schema.

`PlannedQuery` is the boundary [ADR 0004] asks for: malformed planner output
fails here rather than degrading into a subtly wrong search. Nothing in these
tests touches a model — the schema is data, and the LLM call that produces it is
a later issue.
"""

from __future__ import annotations

import dataclasses

import pytest

from mtg_rag.plan.query import PlannedQuery


def test_a_well_formed_query_keeps_both_fields() -> None:
    query = PlannedQuery(query_text="graveyard recursion", purpose="theme payoff")

    assert query.query_text == "graveyard recursion"
    assert query.purpose == "theme payoff"


def test_planned_query_requires_non_empty_text() -> None:
    with pytest.raises(ValueError, match="query_text"):
        PlannedQuery(query_text="", purpose="ramp")


def test_planned_query_requires_non_empty_purpose() -> None:
    # `purpose` travels to curation as its starting hypothesis for the card's
    # role ([ADR 0005]), so an empty one is a silently useless plan.
    with pytest.raises(ValueError, match="purpose"):
        PlannedQuery(query_text="mana rocks", purpose="")


def test_whitespace_only_fields_are_rejected() -> None:
    # A model emitting "  " satisfies a naive truthiness check but says nothing.
    with pytest.raises(ValueError, match="query_text"):
        PlannedQuery(query_text="   ", purpose="ramp")
    with pytest.raises(ValueError, match="purpose"):
        PlannedQuery(query_text="mana rocks", purpose="\t")


def test_planned_query_is_frozen() -> None:
    query = PlannedQuery(query_text="self-mill", purpose="enabler")

    with pytest.raises(dataclasses.FrozenInstanceError):
        query.query_text = "something else"  # type: ignore[misc]


def test_purpose_is_carried_verbatim() -> None:
    # Retrieval treats `purpose` as an opaque label — it is displayed and
    # attached to candidates, never parsed or matched against a vocabulary.
    # Whether it should become a closed set belongs to the curation call.
    query = PlannedQuery(query_text="wraths", purpose="a role nobody enumerated")

    assert query.purpose == "a role nobody enumerated"
