"""Which compute tier, dtype and device this machine supports.

Shared by every module that loads a local torch model — the embedder
([ADR 0012]) and the planner's instruct model ([ADR 0021]) — so the choice is
made here once and imported, not re-derived per package. The constants these
read live in `device_config.py`.
"""

from __future__ import annotations

from typing import Any

import torch

from mtg_rag.device_config import (
    BF16_MIN_COMPUTE_MAJOR,
    TORCH_DEVICE_BY_CAPABILITY,
    TORCH_DTYPE_BY_CAPABILITY,
)


def detect_capability(torch: Any) -> str:
    """Which compute tier `torch` reports for this machine.

    Takes the module as an argument rather than reading the import directly, so
    every branch is testable on one machine: a test fakes Ampere, Turing, MPS
    and CPU in turn without owning any of that hardware.

    Ordered by preference: natively bf16-capable CUDA, then any CUDA, then
    Apple's MPS, then CPU as the floor.

    The bf16 tier is decided on compute capability rather than on
    `torch.cuda.is_bf16_supported()`, which answers True on Turing as well —
    it counts emulation. Emulated bf16 runs, but without a tensor-core path it
    is slower than the float16 those cards do accelerate, so believing that
    call would quietly pick the worse dtype on exactly the hardware this
    project targets.
    """
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        return "cuda-bf16" if major >= BF16_MIN_COMPUTE_MAJOR else "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype() -> str:
    """The widest compute dtype this machine actually supports."""
    return TORCH_DTYPE_BY_CAPABILITY[detect_capability(torch)]


def resolve_device() -> str:
    """The torch device this machine's best backend lives on."""
    return TORCH_DEVICE_BY_CAPABILITY[detect_capability(torch)]
