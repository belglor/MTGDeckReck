"""Tests for parsing the planner's raw model output into a plan.

The boundary [ADR 0022] asks for: the model's raw text becomes a validated
`list[PlannedQuery]` or it raises — never scraped from prose, never degraded to
a partial or default plan. Nothing here touches a model; the text is the input.
"""

from __future__ import annotations

import json

import pytest

from mtg_rag.plan.parse import MalformedPlanError, parse_plan
from mtg_rag.plan.query import PlannedQuery


def _dump(entries: list[dict[str, str]]) -> str:
    return json.dumps(entries)


def test_a_valid_payload_becomes_the_expected_plan() -> None:
    text = _dump(
        [
            {"query_text": "graveyard recursion", "purpose": "theme payoff"},
            {"query_text": "mana rocks", "purpose": "ramp"},
        ]
    )

    assert parse_plan(text) == [
        PlannedQuery(query_text="graveyard recursion", purpose="theme payoff"),
        PlannedQuery(query_text="mana rocks", purpose="ramp"),
    ]


def test_malformed_json_raises() -> None:
    with pytest.raises(MalformedPlanError, match="JSON"):
        parse_plan("{not json")


def test_prose_wrapped_around_json_raises_not_scraped() -> None:
    # The exact failure the schema exists to kill: a phrasing drift must become a
    # validation error, never a query scraped out of surrounding prose (ADR 0004).
    text = 'Here is your plan: [{"query_text": "wraths", "purpose": "removal"}]'

    with pytest.raises(MalformedPlanError):
        parse_plan(text)


def test_a_non_list_top_level_raises() -> None:
    text = _dump([{"query_text": "self-mill", "purpose": "enabler"}])
    text = json.dumps({"queries": json.loads(text)})

    with pytest.raises(MalformedPlanError, match="list"):
        parse_plan(text)


def test_an_empty_list_raises() -> None:
    # A plan with no queries says nothing; it should fail loudly, not run empty.
    with pytest.raises(MalformedPlanError, match="empty"):
        parse_plan("[]")


def test_a_missing_key_raises() -> None:
    text = _dump([{"query_text": "looting"}])  # type: ignore[list-item]

    with pytest.raises(MalformedPlanError, match="purpose"):
        parse_plan(text)


def test_an_extra_key_raises() -> None:
    text = json.dumps([{"query_text": "connive", "purpose": "theme payoff", "weight": "3"}])

    with pytest.raises(MalformedPlanError, match="key"):
        parse_plan(text)


def test_a_non_object_entry_raises() -> None:
    with pytest.raises(MalformedPlanError):
        parse_plan(json.dumps(["graveyard recursion"]))


def test_a_wrong_value_type_raises() -> None:
    text = json.dumps([{"query_text": "ramp", "purpose": 3}])

    with pytest.raises(MalformedPlanError):
        parse_plan(text)


def test_a_whitespace_only_field_raises() -> None:
    # `PlannedQuery.__post_init__` already rejects this; the parser must let it
    # through to raise rather than filtering the entry out.
    text = _dump([{"query_text": "   ", "purpose": "ramp"}])

    with pytest.raises(MalformedPlanError, match="query_text"):
        parse_plan(text)


def test_every_failure_names_what_was_wrong() -> None:
    # The message is the diagnostic; a bare raise leaves the operator guessing.
    with pytest.raises(MalformedPlanError) as excinfo:
        parse_plan("not json at all")

    assert str(excinfo.value)
