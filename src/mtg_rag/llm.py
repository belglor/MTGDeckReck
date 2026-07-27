"""The seam between an app stage and a generative model.

`LLMClient` is what a stage depends on; `QwenChatClient` is the one
implementation ([ADR 0021]). Keeping the model behind a protocol is what lets a
stage be tested with a deterministic fake instead of a multi-GB download — the
same shape the embedder uses for `Encoder` (`src/mtg_rag/embed/encoder.py`).
Shared by the planner and curation ([ADR 0021]) rather than owned by either.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Protocol, cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention

from mtg_rag.device import detect_capability, resolve_device, resolve_torch_dtype
from mtg_rag.device_config import ATTENTION_IMPLEMENTATION
from mtg_rag.llm_config import (
    ENABLE_THINKING,
    MAX_NEW_TOKENS,
    MODEL_ID,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

# transformers ships partial type information, so `from_pretrained` resolves to a
# partially-unknown type that trips pyright strict. Bind the factory classes to
# `Any` here — one explicit boundary, rather than a suppression at each
# `.generate()` / `.decode()` call the loaded objects reach downstream.
_AutoModelForCausalLM: Any = AutoModelForCausalLM
_AutoTokenizer: Any = AutoTokenizer


@contextmanager
def prefer_efficient_attention(*, force: bool) -> Generator[None]:
    """Force SDPA's memory-efficient kernel for one `generate()` call ([#93]).

    Not underscore-prefixed like this module's other helpers: `QwenChatClient`
    stays deliberately untested (a model-loading test means a multi-GB download
    or a tautological mock), so the one piece of real branching logic it
    contains needs to be reachable from `test_llm.py` without either.

    A no-op unless `force`, which `QwenChatClient` only sets on Turing-class
    CUDA — the one tier this is verified on. Forcing it on CPU raises outright
    ("No viable backend"); Ampere+ CUDA and MPS are untested, so they keep
    today's default dispatch.

    On that Turing tier, the default `sdpa` dispatch falls back to the
    quadratic-memory "math" kernel — not because the hardware lacks a cheaper
    one, but because Qwen3's grouped-query attention (16 query heads, 8 KV
    heads) makes `transformers` pass `enable_gqa=True` to
    `scaled_dot_product_attention`, and neither fused kernel in this torch
    build honours that flag, so both refuse the call and only `math` is left
    (`curate/config.py`'s `CURATION_POOL_SIZE` has the measured before/after).
    Verified against `torch==2.11.0+cu128`, `transformers==5.14.1`.

    The fix: make `transformers` take the `repeat_kv` branch it already ships
    for pre-GQA-aware torch instead of the `enable_gqa` fast path, by patching
    `use_gqa_in_sdpa` — an internal helper, not a public setting — to always
    answer False, then restricting SDPA to the now-eligible efficient-attention
    kernel for the duration of one call. The patch is undone before this
    returns either way, so it can never leak into another caller sharing this
    process (the embedder, were it ever invoked before curation's forward pass
    finishes).

    Guarded rather than assumed: if a future `transformers` upgrade removes
    `use_gqa_in_sdpa`, patching it would silently do nothing, so this checks
    first and warns instead of raising — the `LLMClient` seam is shared with
    the planner, which has no need of this optimization and should not break
    because of it.
    """
    if not force:
        yield
        return
    if not hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        warnings.warn(
            "transformers.integrations.sdpa_attention.use_gqa_in_sdpa is gone; "
            "the issue #93 prefill workaround no longer applies, so curation "
            "falls back to the slower default attention kernel.",
            stacklevel=2,
        )
        yield
        return

    original_use_gqa_in_sdpa = sdpa_attention.use_gqa_in_sdpa
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key, value: False
    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            yield
    finally:
        sdpa_attention.use_gqa_in_sdpa = original_use_gqa_in_sdpa


class LLMClient(Protocol):
    """What a stage needs from a generative model: raw text back.

    One method by design. A caller prompts the model and gets a string; parsing
    and validating that string into a typed result is a separate boundary
    (planner: [ADR 0022]), so the client returns text, not a typed result.
    Adding a method no caller uses would be scaffolding no stage reaches for.
    """

    def complete(self, *, system: str, user: str) -> str: ...


class QwenChatClient:
    """`Qwen/Qwen3-1.7B`, run locally ([ADR 0021]).

    Deliberately untested, like `QwenEncoder`: a thin adapter whose only
    behavior is configuration, so a test would mean a multi-GB download or
    mocking the library into a tautology. It is guarded structurally instead —
    callers depend on `LLMClient`, and their tests pass a deterministic fake.
    The one piece of real branching logic it reaches for,
    `prefer_efficient_attention`, is tested on its own, without a model.
    """

    def __init__(self, *, device: str | None = None) -> None:
        # dtype and device follow the detected hardware rather than assuming the
        # machine this was written on, reusing the embedder's selection so the
        # fp16 / bf16 / MPS / CPU logic lives in one place ([ADR 0021]).
        tokenizer: Any = _AutoTokenizer.from_pretrained(MODEL_ID)
        model: Any = _AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=resolve_torch_dtype(),
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )

        self._tokenizer = tokenizer
        self._model = model.to(device or resolve_device()).eval()
        # Turing-class CUDA only — see `prefer_efficient_attention` ([#93]).
        self._force_efficient_attention = detect_capability(torch) == "cuda"

    def complete(self, *, system: str, user: str) -> str:
        """Generate the model's raw completion for a system + user prompt."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # `enable_thinking=False` switches off Qwen3's reasoning preamble so the
        # completion is the answer alone ([ADR 0021]); left on, it would have to
        # be stripped before parsing.
        prompt: Any = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        inputs: Any = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with prefer_efficient_attention(force=self._force_efficient_attention):
            generated: Any = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )

        # `generate` returns the prompt followed by the completion; the caller
        # wants the completion alone, so drop the prompt's tokens before decoding.
        prompt_length: int = inputs["input_ids"].shape[1]
        completion_ids: Any = generated[0][prompt_length:]
        return cast("str", self._tokenizer.decode(completion_ids, skip_special_tokens=True))
