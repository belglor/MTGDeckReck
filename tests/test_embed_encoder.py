"""Tests for the encoder's protocol boundary.

`QwenEncoder` itself stays deliberately untested — it is a thin configuration
adapter, so a test would mean either a 1.2 GB download or mocking the library
into a tautology. What is worth guarding is that the pipeline depends on the
`Encoder` protocol, never on the concrete model, so a deterministic fake can
stand in. The hardware detection the encoder relies on is tested in
`test_device.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from mtg_rag.embed.config import (
    DOCUMENT_BATCH_SIZE,
    QUERY_BATCH_SIZE,
)
from mtg_rag.embed.encoder import Encoder


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
