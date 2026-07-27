"""Tests for the curation entry point that ties the stage together.

`curate` reads the format template ([ADR 0025]), builds the prompt from it and
the retrieved pool, calls the injected `LLMClient` ([ADR 0021]), and validates
the output into a `list[CuratedCard]` ([ADR 0024]). On malformed output it
retries exactly once, then raises — never a second retry, never a degraded
recommendation, even though nothing downstream but the user consumes it.

The `LLMClient` is faked with canned text, so nothing here touches a model, the
network, or a weights download: the whole retry policy is exercised against a
deterministic collaborator.
"""

from __future__ import annotations

import json

import pytest

from mtg_rag.curate.config import MAX_CURATION_RETRIES
from mtg_rag.curate.curation import curate
from mtg_rag.curate.parse import MalformedRecommendationError
from mtg_rag.curate.prompt import CurationCard
from mtg_rag.curate.recommendation import CuratedCard

_REQUEST = "a spooky graveyard deck that mills itself"

_CARDS = (
    CurationCard(
        oracle_id="abc-123",
        name="Altar of Dementia",
        mana_cost="{2}",
        type_line="Artifact",
        oracle_text="Sacrifice a creature: Target player mills cards equal to its power.",
        flavor_text="",
        purposes=("theme payoff",),
    ),
    CurationCard(
        oracle_id="def-456",
        name="Sol Ring",
        mana_cost="{1}",
        type_line="Artifact",
        oracle_text="{T}: Add {C}{C}.",
        flavor_text="",
        purposes=("ramp",),
    ),
)

_VALID = json.dumps(
    [
        {
            "oracle_id": "abc-123",
            "role": "theme payoff",
            "rationale": "Sacrifices creatures to mill, which is the deck's whole plan.",
        },
        {
            "oracle_id": "def-456",
            "role": "ramp",
            "rationale": "Two colourless mana a turn early, accelerating into the payoffs.",
        },
    ]
)
_MALFORMED = "here you go: not json at all"


class _FakeClient:
    """A canned `LLMClient`: hands back queued responses and counts its calls.

    The count is the retry policy's witness — a test asserts curation calls the
    model at most twice ([ADR 0024]).
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        return self._responses.pop(0)


def test_happy_path_returns_the_parsed_recommendation_in_one_call() -> None:
    client = _FakeClient([_VALID])

    result = curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert result == [
        CuratedCard(
            oracle_id="abc-123",
            role="theme payoff",
            rationale="Sacrifices creatures to mill, which is the deck's whole plan.",
        ),
        CuratedCard(
            oracle_id="def-456",
            role="ramp",
            rationale="Two colourless mana a turn early, accelerating into the payoffs.",
        ),
    ]
    assert client.calls == 1


def test_malformed_then_valid_returns_on_the_retry() -> None:
    client = _FakeClient([_MALFORMED, _VALID])

    result = curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert [card.oracle_id for card in result] == ["abc-123", "def-456"]
    assert client.calls == 2


def test_malformed_twice_raises_and_never_calls_a_third_time() -> None:
    client = _FakeClient([_MALFORMED, _MALFORMED])

    with pytest.raises(MalformedRecommendationError):
        curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert client.calls == 2


def test_total_attempts_follow_the_configured_retry_cap() -> None:
    # The cap is a named constant ([ADR 0024]), not a literal in the call: with
    # every attempt malformed, the model is called exactly one more time than the
    # cap allows retries. Ties the test to the constant, so a drift in either
    # moves them together.
    client = _FakeClient([_MALFORMED] * (MAX_CURATION_RETRIES + 1))

    with pytest.raises(MalformedRecommendationError):
        curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert client.calls == MAX_CURATION_RETRIES + 1


def test_the_named_format_template_and_pool_are_read_and_prompted() -> None:
    # `curate` resolves `format_name` to a template file and feeds its text to
    # the prompt ([ADR 0025]). Commander's heading is a stable witness that the
    # file was read; the request and a card name witness the pool riding along.
    client = _FakeClient([_VALID])

    curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert "# Commander" in client.last_system
    assert _REQUEST in client.last_user
    assert "Altar of Dementia" in client.last_user


def test_an_id_outside_the_given_pool_is_malformed() -> None:
    # The closed-vocabulary check needs the pool, and `curate` is what has it
    # ([ADR 0024]). A well-formed entry naming a card that was never a candidate
    # is a validation failure, so it burns the retry and then raises rather than
    # reaching the user attached to an unmoored rationale.
    invented = json.dumps(
        [{"oracle_id": "not-in-pool", "role": "ramp", "rationale": "Invented card."}]
    )
    client = _FakeClient([invented, invented])

    with pytest.raises(MalformedRecommendationError):
        curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert client.calls == 2


def test_selecting_no_candidate_is_a_valid_recommendation() -> None:
    # An empty pick is a real answer, not a malformed one ([ADR 0024]): nothing
    # in the pool belonged. It must not trigger the retry.
    client = _FakeClient(["[]"])

    result = curate(_REQUEST, format_name="commander", cards=_CARDS, client=client)

    assert result == []
    assert client.calls == 1
