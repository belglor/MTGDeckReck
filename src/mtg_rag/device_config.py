"""Hardware-dependent model-loading constants — values only.

Shared by every module that loads a local torch model: the embedder
([ADR 0012]) and the planner's instruct model ([ADR 0021]) run on the same
single GPU, so the CUDA-fp16 / bf16 / MPS / CPU choice — and the attention
kernel — are settled here once rather than re-derived per package. The logic
that reads these lives in `device.py`.
"""

from __future__ import annotations

from collections.abc import Mapping

#: CUDA compute-capability major version at which bfloat16 becomes *native*:
#: Ampere, sm_80. Below it torch still reports bf16 as supported, because
#: `torch.cuda.is_bf16_supported()` counts emulation — measured True on a
#: Turing RTX 2070 (sm_75), where bf16 has no tensor-core path and is slower
#: than plain float16. The tier is decided on this number for that reason.
BF16_MIN_COMPUTE_MAJOR = 8

#: Compute dtype per detected device capability — the hardware assumption made
#: explicit, rather than one machine's answer hardcoded at the call site.
#: bfloat16 needs Ampere or newer (sm_80+); Turing (sm_75), the RTX 2070 class
#: this was first written for, offers float16 only. CPU gets float32: half
#: precision there is slow and unevenly supported, and the routed Linux wheel is
#: a CPU build, so that path is reachable rather than hypothetical.
TORCH_DTYPE_BY_CAPABILITY: Mapping[str, str] = {
    "cuda-bf16": "bfloat16",
    "cuda": "float16",
    "mps": "float16",
    "cpu": "float32",
}

#: Torch device string per detected capability. The bf16 and plain CUDA tiers
#: differ only in dtype; both place the model on `cuda`. The embedder delegates
#: placement to sentence-transformers, but a raw `transformers` model must be
#: told where to go, so the planner reads this.
TORCH_DEVICE_BY_CAPABILITY: Mapping[str, str] = {
    "cuda-bf16": "cuda",
    "cuda": "cuda",
    "mps": "mps",
    "cpu": "cpu",
}

#: Attention kernel. `sdpa` ships with torch, needs no extra package, and runs
#: on every backend below. flash-attention-2 is deliberately not requested: it
#: would need `flash-attn` installed and Ampere or newer, and it is not a
#: dependency this project carries.
#:
#: Which *backend* `sdpa` picks underneath this string is a separate question
#: ([#93]), answered once on the target 8 GB RTX 2070 (`torch==2.11.0+cu128`):
#: `FLASH_ATTENTION` is unavailable — this torch build was not compiled with
#: it, independent of hardware — and `CUDNN_ATTENTION` refuses Turing outright
#: (needs sm80+; this is sm_75). Both are negative results, not gaps to revisit
#: without a torch or hardware change. `EFFICIENT_ATTENTION` is available and
#: is forced for curation's prefill, worked around a grouped-query-attention
#: mismatch that otherwise blocks it too — the mechanism, and why this lives
#: in `llm.py` rather than as a constant here, is `prefer_efficient_attention`
#: next to `QwenChatClient`.
ATTENTION_IMPLEMENTATION = "sdpa"
