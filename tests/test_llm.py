"""Tests for the shared chat-model seam.

`QwenChatClient` itself stays deliberately untested — it is a thin
configuration adapter, so a test would mean either a multi-GB download or
mocking the library into a tautology. What is worth guarding is that a caller
depends on the `LLMClient` protocol, never on the concrete model, so a
deterministic fake can stand in with no weights ([ADR 0021]).

`prefer_efficient_attention` is the one piece of real branching logic in this
module, and needs neither a model nor a GPU to exercise: entering and exiting
`sdpa_kernel` is plain Python bookkeeping, and what these tests check is
whether `transformers`' internal `use_gqa_in_sdpa` gets patched and restored
correctly around it ([#93]).
"""

from __future__ import annotations

import pytest
import torch
from transformers.integrations import sdpa_attention

from mtg_rag.llm import LLMClient, prefer_efficient_attention


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


def test_unforced_leaves_use_gqa_in_sdpa_untouched() -> None:
    # Every tier but Turing-class CUDA ([`QwenChatClient.__init__`]) keeps
    # today's default SDPA dispatch, so `force=False` must be a true no-op.
    original = sdpa_attention.use_gqa_in_sdpa

    with prefer_efficient_attention(force=False):
        assert sdpa_attention.use_gqa_in_sdpa is original

    assert sdpa_attention.use_gqa_in_sdpa is original


def test_forced_patches_use_gqa_in_sdpa_and_restores_it_after() -> None:
    # The patch is what makes `transformers` take the `repeat_kv` branch
    # instead of `enable_gqa=True`, which is what makes EFFICIENT_ATTENTION
    # eligible for Qwen3's grouped-query attention in the first place.
    original = sdpa_attention.use_gqa_in_sdpa
    dummy = torch.empty(0)

    with prefer_efficient_attention(force=True):
        assert sdpa_attention.use_gqa_in_sdpa is not original
        assert sdpa_attention.use_gqa_in_sdpa(None, dummy, dummy) is False

    assert sdpa_attention.use_gqa_in_sdpa is original


def test_forced_restores_use_gqa_in_sdpa_even_if_the_body_raises() -> None:
    # `complete`'s `.generate()` call lives inside this context manager; a
    # generation failure must not leave the patch dangling for whichever
    # caller — curation, the planner, or a future embedder call — runs next.
    original = sdpa_attention.use_gqa_in_sdpa

    with pytest.raises(RuntimeError, match="boom"), prefer_efficient_attention(force=True):
        raise RuntimeError("boom")

    assert sdpa_attention.use_gqa_in_sdpa is original


def test_missing_use_gqa_in_sdpa_warns_and_falls_back_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A future transformers upgrade could remove or rename this internal
    # helper. The shared `LLMClient` seam — the planner depends on it too,
    # with no need of this optimization — must keep working even if the
    # issue #93 workaround stops applying; it should just get slower, not
    # break, and it should say so rather than silently doing nothing.
    monkeypatch.delattr(sdpa_attention, "use_gqa_in_sdpa")

    with (
        pytest.warns(UserWarning, match="use_gqa_in_sdpa is gone"),
        prefer_efficient_attention(force=True),
    ):
        pass
