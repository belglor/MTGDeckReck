"""Tests for the shared chat-model seam.

`QwenChatClient` itself stays deliberately untested — it is a thin
configuration adapter, so a test would mean either a multi-GB download or
mocking the library into a tautology. What is worth guarding is that a caller
depends on the `LLMClient` protocol, never on the concrete model, so a
deterministic fake can stand in with no weights ([ADR 0021]).
"""

from __future__ import annotations

from mtg_rag.llm import LLMClient


class FakeLLMClient:
    """A deterministic stand-in — the reason a caller depends on a protocol.

    It echoes the prompt it was handed, which is enough to assert a caller used
    it correctly without loading a model.
    """

    def __init__(self, reply: str = "[]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def test_a_deterministic_fake_satisfies_the_llm_client_protocol() -> None:
    # A caller depends on `LLMClient`, never on `QwenChatClient` — that is what
    # lets it be exercised without a model. This assignment is the assertion:
    # if the protocol grew a method only the real model could provide, it would
    # stop typechecking here rather than where it was needed.
    client: LLMClient = FakeLLMClient(reply='[{"query_text": "wraths", "purpose": "removal"}]')

    completion = client.complete(system="You are a planner.", user="A spooky graveyard deck.")

    assert completion == '[{"query_text": "wraths", "purpose": "removal"}]'


def test_the_fake_records_the_prompt_it_was_given() -> None:
    # The seam carries a system and a user prompt as separate arguments; a fake
    # can capture both, which is what lets a caller's test assert on the prompt
    # it assembled without a model in the loop.
    client = FakeLLMClient()

    client.complete(system="rules", user="request")

    assert client.calls == [("rules", "request")]
