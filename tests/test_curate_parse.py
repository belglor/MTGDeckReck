"""Tests for parsing the curation model's raw output into a validated recommendation.

The boundary [ADR 0024] asks for: the model's raw text becomes a validated
`list[CuratedCard]` or it raises — never scraped from prose, never degraded to
a partial recommendation. The closed-vocabulary rule is checked here too: every
returned `oracle_id` must be a member of the retrieved pool. Nothing here
touches a model; the text and pool ids are the input.
"""

from __future__ import annotations

import json

import pytest

from mtg_rag.curate.parse import MalformedRecommendationError, parse_recommendation
from mtg_rag.curate.recommendation import CuratedCard

POOL_IDS = frozenset({"abc123", "def456"})


def _dump(entries: list[dict[str, str]]) -> str:
    return json.dumps(entries)


def test_a_valid_payload_becomes_the_expected_recommendation() -> None:
    text = _dump(
        [
            {"oracle_id": "abc123", "role": "payoff", "rationale": "recurs lands from graveyard"},
            {"oracle_id": "def456", "role": "ramp", "rationale": "fixes and ramps mana"},
        ]
    )

    assert parse_recommendation(text, pool_ids=POOL_IDS) == [
        CuratedCard(oracle_id="abc123", role="payoff", rationale="recurs lands from graveyard"),
        CuratedCard(oracle_id="def456", role="ramp", rationale="fixes and ramps mana"),
    ]


def test_an_empty_list_is_a_valid_recommendation() -> None:
    # Unlike the planner's plan, an empty recommendation says something
    # legitimate: curation may select no candidate from the pool.
    assert parse_recommendation("[]", pool_ids=POOL_IDS) == []


def test_malformed_json_raises() -> None:
    with pytest.raises(MalformedRecommendationError, match="JSON"):
        parse_recommendation("{not json", pool_ids=POOL_IDS)


def test_prose_wrapped_around_json_raises_not_scraped() -> None:
    # The exact failure the schema exists to kill: a phrasing drift must become
    # a validation error, never a card scraped out of surrounding prose.
    text = 'Here: [{"oracle_id": "abc123", "role": "payoff", "rationale": "recurs lands"}]'

    with pytest.raises(MalformedRecommendationError):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_a_non_list_top_level_raises() -> None:
    text = json.dumps(
        {"cards": [{"oracle_id": "abc123", "role": "payoff", "rationale": "recurs lands"}]}
    )

    with pytest.raises(MalformedRecommendationError, match="list"):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_a_missing_key_raises() -> None:
    text = _dump([{"oracle_id": "abc123", "role": "payoff"}])  # type: ignore[list-item]

    with pytest.raises(MalformedRecommendationError, match="rationale"):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_an_extra_key_raises() -> None:
    text = json.dumps(
        [{"oracle_id": "abc123", "role": "payoff", "rationale": "recurs lands", "score": "9"}]
    )

    with pytest.raises(MalformedRecommendationError, match="key"):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_a_non_object_entry_raises() -> None:
    with pytest.raises(MalformedRecommendationError):
        parse_recommendation(json.dumps(["abc123"]), pool_ids=POOL_IDS)


def test_a_wrong_value_type_raises() -> None:
    text = json.dumps([{"oracle_id": "abc123", "role": "payoff", "rationale": 3}])

    with pytest.raises(MalformedRecommendationError):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_a_whitespace_only_field_raises() -> None:
    # `CuratedCard.__post_init__` already rejects this; the parser must let it
    # through to raise rather than filtering the entry out.
    text = _dump([{"oracle_id": "abc123", "role": "   ", "rationale": "recurs lands"}])

    with pytest.raises(MalformedRecommendationError, match="role"):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_an_out_of_pool_oracle_id_raises() -> None:
    # An invented id hydrates to nothing (or the wrong card); it must never be
    # accepted just because the rest of the entry is well-formed.
    text = _dump([{"oracle_id": "invented999", "role": "payoff", "rationale": "recurs lands"}])

    with pytest.raises(MalformedRecommendationError, match="invented999"):
        parse_recommendation(text, pool_ids=POOL_IDS)


def test_an_omitted_pool_id_does_not_raise() -> None:
    # "def456" is in the pool but the model never returns it — that is fine,
    # not every candidate makes the deck (ADR 0024).
    text = _dump([{"oracle_id": "abc123", "role": "payoff", "rationale": "recurs lands"}])

    result = parse_recommendation(text, pool_ids=POOL_IDS)

    assert result == [CuratedCard(oracle_id="abc123", role="payoff", rationale="recurs lands")]


def test_every_failure_names_what_was_wrong() -> None:
    # The message is the diagnostic; a bare raise leaves the operator guessing.
    with pytest.raises(MalformedRecommendationError) as excinfo:
        parse_recommendation("not json at all", pool_ids=POOL_IDS)

    assert str(excinfo.value)
