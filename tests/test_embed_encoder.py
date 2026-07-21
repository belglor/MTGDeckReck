"""Tests for the encoder's hardware detection.

`QwenEncoder` itself stays deliberately untested — it is a thin configuration
adapter. The branch worth testing is which dtype it asks for, because that
decision is hardware-dependent and every wrong answer fails quietly rather than
loudly: float16 on CPU is slow and unevenly supported, and Turing cannot do
bfloat16 at all.

`detect_capability` takes the torch module as an argument rather than importing
it, so this runs on a machine with no torch — which is every machine that has
not opted into the `embed` extra, CI included.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mtg_rag.embed.config import TORCH_DTYPE_BY_CAPABILITY
from mtg_rag.embed.encoder import detect_capability


def _torch(*, cuda: bool, bf16: bool = False, mps: bool = False) -> Any:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_ampere_and_newer_get_bfloat16() -> None:
    assert detect_capability(_torch(cuda=True, bf16=True)) == "cuda-bf16"
    assert TORCH_DTYPE_BY_CAPABILITY["cuda-bf16"] == "bfloat16"


def test_turing_falls_back_to_float16() -> None:
    # sm_75 has no bf16 — the RTX 2070 path the model was originally chosen for.
    assert detect_capability(_torch(cuda=True, bf16=False)) == "cuda"
    assert TORCH_DTYPE_BY_CAPABILITY["cuda"] == "float16"


def test_apple_mps_is_used_when_there_is_no_cuda() -> None:
    assert detect_capability(_torch(cuda=False, mps=True)) == "mps"
    assert TORCH_DTYPE_BY_CAPABILITY["mps"] == "float16"


def test_cpu_only_machines_get_float32() -> None:
    # Reachable rather than hypothetical: the routed Linux wheel is a CPU build.
    assert detect_capability(_torch(cuda=False, mps=False)) == "cpu"
    assert TORCH_DTYPE_BY_CAPABILITY["cpu"] == "float32"


def test_a_torch_build_without_the_mps_backend_is_not_an_error() -> None:
    # Older torch builds have no `torch.backends.mps` attribute at all, so
    # probing for it must not raise on the way to the CPU floor.
    torch: Any = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        backends=SimpleNamespace(),
    )

    assert detect_capability(torch) == "cpu"
