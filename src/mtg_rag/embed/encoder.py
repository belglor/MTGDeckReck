"""Turning text into vectors — the only module here that touches a model.

`Encoder` is what the rest of the pipeline depends on; `QwenEncoder` is the one
implementation ([ADR 0012]). Everything else composes text or moves vectors
around, so keeping the model behind a protocol is what lets the pipeline be
tested with a deterministic fake instead of a 1.2 GB download.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

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


def _resolve_torch_dtype() -> str:
    """The widest compute dtype this machine actually supports."""
    return TORCH_DTYPE_BY_CAPABILITY[detect_capability(torch)]


class QwenEncoder:
    """`Qwen/Qwen3-Embedding-0.6B`, run locally ([ADR 0012]).

    Deliberately untested: it is a thin adapter whose only behavior is
    configuration, so a test would mean either a 1.2 GB download or mocking the
    library into a tautology. It is guarded structurally instead — the pipeline
    depends on `Encoder`, and the tests pass a deterministic fake.
    """

    def __init__(self, *, device: str | None = None) -> None:
        # Annotated `Any` deliberately: `encode_document` and `encode_query` are
        # only partially annotated upstream, and letting that leak would spread
        # `Unknown` through every caller of this class. One explicit boundary
        # here is better than suppressions at each use.
        #
        # The dtype follows the hardware rather than assuming the machine this
        # was written on. Note there is no `padding_side="left"`: it appears in
        # the model card's raw-transformers example and gets cargo-culted, but
        # sentence-transformers' last-token pooling reads the attention mask and
        # is padding-side agnostic.
        model: Any = SentenceTransformer(
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
