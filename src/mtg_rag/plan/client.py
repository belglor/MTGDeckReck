"""The seam between the planner and a generative model.

`LLMClient` is what the planner depends on; `QwenPlannerClient` is the one
implementation ([ADR 0021]). Keeping the model behind a protocol is what lets
the planner be tested with a deterministic fake instead of a multi-GB download —
the same shape the embedder uses for `Encoder` (`src/mtg_rag/embed/encoder.py`).
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

from mtg_rag.device import resolve_device, resolve_torch_dtype
from mtg_rag.device_config import ATTENTION_IMPLEMENTATION
from mtg_rag.plan.config import (
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


class LLMClient(Protocol):
    """What the planner needs from a generative model: raw text back.

    One method by design. The planner prompts the model and gets a string;
    parsing and validating that string into `PlannedQuery` is a separate
    boundary ([ADR 0022]), so the client returns text, not a typed plan. Adding
    a method no caller uses would be scaffolding the planner never reaches for.
    """

    def complete(self, *, system: str, user: str) -> str: ...


class QwenPlannerClient:
    """`Qwen/Qwen3-1.7B`, run locally ([ADR 0021]).

    Deliberately untested, like `QwenEncoder`: a thin adapter whose only
    behavior is configuration, so a test would mean a multi-GB download or
    mocking the library into a tautology. It is guarded structurally instead —
    the planner depends on `LLMClient`, and the tests pass a deterministic fake.
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
