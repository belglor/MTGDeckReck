"""Turning text into vectors — the only module here that touches a model.

`Encoder` is what the rest of the pipeline depends on; `QwenEncoder` is the one
implementation ([ADR 0012]). Everything else composes text or moves vectors
around, so keeping the model behind a protocol is what lets the pipeline be
tested with a deterministic fake instead of a 1.2 GB download.

**Importing this module must not import torch.** `sentence_transformers` is a
~2.5 GB dependency needed only by the machine running `just embed`, so it lives
behind the `embed` extra and is loaded when a `QwenEncoder` is constructed
rather than at module scope. That keeps `python -m mtg_rag.embed` importable —
and the test suite runnable — on an install that has no model at all.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from mtg_rag.embed.config import (
    ATTENTION_IMPLEMENTATION,
    BF16_MIN_COMPUTE_MAJOR,
    DOCUMENT_BATCH_SIZE,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    QUERY_BATCH_SIZE,
    TORCH_DTYPE_BY_CAPABILITY,
)


class Encoder(Protocol):
    """What the pipeline needs from an embedding model.

    Documents and queries are separate calls rather than one `encode` with a
    flag, because the model is asymmetric by design: its document prompt is the
    empty string and its query prompt is an `"Instruct: …"` preamble. Both sides
    must honour that or the geometry silently mismatches, so the asymmetry lives
    in the API instead of a comment ([ADR 0012]).
    """

    dim: int

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
    ) -> NDArray[np.float32]: ...

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int = QUERY_BATCH_SIZE
    ) -> NDArray[np.float32]: ...


def _load_sentence_transformer() -> Any:
    """Import `SentenceTransformer` on demand.

    Imported by name rather than with a `from ... import`, because the package
    is deliberately absent from the default install: a static import would be
    unresolvable to the typechecker on exactly the installs this project
    expects, and suppressing that would spread `Unknown` through everything
    downstream of the model handle. Going through `importlib` keeps the boundary
    a single, explicitly-typed `Any` and lets a missing extra explain itself.
    """
    try:
        module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
        raise ModuleNotFoundError(
            "The embedding model is an optional dependency, and it is not installed. "
            "Run `uv sync --extra embed` (this pulls torch, roughly 2.5 GB)."
        ) from exc
    loaded: Any = module.SentenceTransformer
    return loaded


def detect_capability(torch: Any) -> str:
    """Which compute tier `torch` reports for this machine.

    Takes the module rather than importing it, so the branch that decides the
    dtype is testable on a machine with no torch at all — which is every machine
    that has not opted into the `embed` extra, including CI.

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


def _resolve_torch_dtype() -> str:
    """The widest compute dtype this machine actually supports."""
    torch: Any = importlib.import_module("torch")
    return TORCH_DTYPE_BY_CAPABILITY[detect_capability(torch)]


class QwenEncoder:
    """`Qwen/Qwen3-Embedding-0.6B`, run locally ([ADR 0012]).

    Deliberately untested: it is a thin adapter whose only behavior is
    configuration, so a test would mean either a 1.2 GB download or mocking the
    library into a tautology. It is guarded structurally instead — the pipeline
    depends on `Encoder`, and the tests pass a deterministic fake.
    """

    def __init__(self, *, device: str | None = None) -> None:
        # Deferred so that importing this module never pulls torch; see the
        # module docstring and `test_embed_imports.py`, which protects it.
        sentence_transformer = _load_sentence_transformer()

        # The dtype follows the hardware rather than assuming the machine this
        # was written on. Note there is no `padding_side="left"`: it appears in
        # the model card's raw-transformers example and gets cargo-culted, but
        # sentence-transformers' last-token pooling reads the attention mask and
        # is padding-side agnostic.
        model: Any = sentence_transformer(
            MODEL_ID,
            device=device,
            model_kwargs={
                "torch_dtype": _resolve_torch_dtype(),
                "attn_implementation": ATTENTION_IMPLEMENTATION,
            },
        )
        model.max_seq_length = MAX_SEQ_LENGTH

        self._model = model
        self.dim = EMBEDDING_DIM

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int = DOCUMENT_BATCH_SIZE
    ) -> NDArray[np.float32]:
        return self._encode(self._model.encode_document, texts, batch_size)

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int = QUERY_BATCH_SIZE
    ) -> NDArray[np.float32]:
        return self._encode(self._model.encode_query, texts, batch_size)

    def _encode(self, encode: Any, texts: Sequence[str], batch_size: int) -> NDArray[np.float32]:
        """Run one of the model's encode methods and normalize the result.

        `normalize_embeddings=True` is not the library default, and cosine
        distance assumes unit norm — without it the store would rank against
        vectors of varying magnitude.
        """
        vectors: Any = encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return cast("NDArray[np.float32]", np.asarray(vectors, dtype=np.float32))
