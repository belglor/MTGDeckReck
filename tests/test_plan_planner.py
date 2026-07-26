"""Tests for the planner entry point that ties the stage together.

`plan` reads the format template, builds the prompt ([ADR 0023]), calls the
injected `LLMClient` ([ADR 0021]), and validates the output into a
`list[PlannedQuery]` ([ADR 0004]). On malformed output it retries exactly once,
then raises ([ADR 0022]) — never a second retry, never a degraded plan.

The `LLMClient` is faked with canned text, so nothing here touches a model, the
network, or a weights download: the whole retry policy is exercised against a
deterministic collaborator.
"""

from __future__ import annotations

import json

import pytest

from mtg_rag.plan.config import MAX_PLAN_RETRIES
from mtg_rag.plan.parse import MalformedPlanError
from mtg_rag.plan.planner import plan
from mtg_rag.plan.query import PlannedQuery

_REQUEST = "a spooky graveyard deck that mills itself"

_VALID = json.dumps(
    [
        {"query_text": "sacrifice for value", "purpose": "theme payoff"},
        {"query_text": "mana rocks", "purpose": "ramp"},
    ]
)
_MALFORMED = "here you go: not json at all"


class _FakeClient:
    """A canned `LLMClient`: hands back queued responses and counts its calls.

    The count is the retry policy's witness — a test asserts the planner calls
    the model at most twice ([ADR 0022]).
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


def test_happy_path_returns_the_parsed_plan_in_one_call() -> None:
    client = _FakeClient([_VALID])

    result = plan(_REQUEST, format_name="commander", client=client)

    assert result == [
        PlannedQuery(query_text="sacrifice for value", purpose="theme payoff"),
        PlannedQuery(query_text="mana rocks", purpose="ramp"),
    ]
    assert client.calls == 1


def test_malformed_then_valid_returns_on_the_retry() -> None:
    client = _FakeClient([_MALFORMED, _VALID])

    result = plan(_REQUEST, format_name="commander", client=client)

    assert [q.query_text for q in result] == ["sacrifice for value", "mana rocks"]
    assert client.calls == 2


def test_malformed_twice_raises_and_never_calls_a_third_time() -> None:
    client = _FakeClient([_MALFORMED, _MALFORMED])

    with pytest.raises(MalformedPlanError):
        plan(_REQUEST, format_name="commander", client=client)

    assert client.calls == 2


def test_total_attempts_follow_the_configured_retry_cap() -> None:
    # The cap is a named constant ([ADR 0022]), not a literal in the call: with
    # every attempt malformed, the model is called exactly one more time than the
    # cap allows retries. Ties the test to the constant, so a drift in either
    # moves them together.
    client = _FakeClient([_MALFORMED] * (MAX_PLAN_RETRIES + 1))

    with pytest.raises(MalformedPlanError):
        plan(_REQUEST, format_name="commander", client=client)

    assert client.calls == MAX_PLAN_RETRIES + 1


def test_the_named_format_template_is_read_and_prompted() -> None:
    # The read this issue adds: `plan` resolves `format_name` to a template file
    # and feeds its text to the prompt. Commander's heading is a stable witness
    # that the file was read and handed through ([ADR 0023]).
    client = _FakeClient([_VALID])

    plan(_REQUEST, format_name="commander", client=client)

    assert "# Commander" in client.last_system
    assert _REQUEST in client.last_user
