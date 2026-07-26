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
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from mtg_rag.device import resolve_torch_dtype
from mtg_rag.device_config import ATTENTION_IMPLEMENTATION
from mtg_rag.embed.config import (
    DOCUMENT_BATCH_SIZE,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    QUERY_BATCH_SIZE,
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
                "torch_dtype": resolve_torch_dtype(),
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
