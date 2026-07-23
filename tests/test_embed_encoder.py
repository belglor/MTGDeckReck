"""Tests for the encoder's hardware detection and its protocol boundary.

`QwenEncoder` itself stays deliberately untested — it is a thin configuration
adapter, so a test would mean either a 1.2 GB download or mocking the library
into a tautology. The branch worth testing is which dtype it asks for, because
that decision is hardware-dependent and every wrong answer fails quietly rather
than loudly: float16 on CPU is slow and unevenly supported, and Turing cannot do
bfloat16 at all.

`detect_capability` takes the torch module as an argument rather than reading
the import directly, which is what lets one machine exercise every branch —
Ampere, Turing, MPS and CPU are all faked here.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mtg_rag.embed.config import (
    DOCUMENT_BATCH_SIZE,
    QUERY_BATCH_SIZE,
    TORCH_DTYPE_BY_CAPABILITY,
)
from mtg_rag.embed.encoder import Encoder, detect_capability


def _torch(*, cuda: bool, compute: tuple[int, int] = (7, 5), mps: bool = False) -> Any:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_capability=lambda: compute,
            # Present, and deliberately answering True even for Turing below —
            # this is the trap `detect_capability` must not fall into.
            is_bf16_supported=lambda: True,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_ampere_and_newer_get_bfloat16() -> None:
    assert detect_capability(_torch(cuda=True, compute=(8, 0))) == "cuda-bf16"
    assert TORCH_DTYPE_BY_CAPABILITY["cuda-bf16"] == "bfloat16"


def test_turing_falls_back_to_float16_despite_torch_claiming_bf16_support() -> None:
    # Measured on a real RTX 2070: torch.cuda.is_bf16_supported() returns True
    # on sm_75 because it counts *emulation*. Emulated bf16 has no tensor-core
    # path and is slower than the float16 Turing does accelerate, so trusting
    # that call would pick the worse dtype on the hardware this targets.
    assert detect_capability(_torch(cuda=True, compute=(7, 5))) == "cuda"
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


# --- the protocol boundary --------------------------------------------------


class FakeEncoder:
    """A deterministic stand-in — the reason the pipeline depends on a protocol.

    Vector length is the only signal, which is enough to make retrieval
    assertions predictable without downloading 1.2 GB of weights.
    """

    def __init__(self) -> None:
        self.dim = 3

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
    ) -> NDArray[np.float32]:
        return np.array([[float(len(text)), 0.0, 0.0] for text in texts], dtype=np.float32)

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int = QUERY_BATCH_SIZE
    ) -> NDArray[np.float32]:
        return np.array([[0.0, float(len(text)), 0.0] for text in texts], dtype=np.float32)


def test_a_deterministic_fake_satisfies_the_encoder_protocol() -> None:
    # The pipeline depends on `Encoder`, never on `QwenEncoder` — that is what
    # lets it be exercised without a model. This assignment is the assertion:
    # if the protocol ever grew something only the real model could provide, it
    # would stop typechecking here rather than at the point someone needed it.
    encoder: Encoder = FakeEncoder()

    documents = encoder.encode_documents(["ab", "cde"])

    assert encoder.dim == 3
    assert documents.shape == (2, 3)
    assert documents.dtype == np.float32
