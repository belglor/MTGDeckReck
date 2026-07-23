"""Guards the boundary that keeps torch out of module-import time.

`sentence_transformers` and `torch` are always installed now, but a torch import
still costs seconds. Nothing in `mtg_rag.embed` may import them at module scope,
or `python -m mtg_rag.ingest` and most of the test suite — neither of which
touches a model — would pay that cost on every import.

This is the regression this file exists for: hoisting the import out of
`QwenEncoder.__init__` up to module scope is a natural-looking tidy-up that
would make the whole package slow to import.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from mtg_rag.embed.config import DOCUMENT_BATCH_SIZE, QUERY_BATCH_SIZE
from mtg_rag.embed.encoder import Encoder

# Run in a fresh interpreter: asserting on this process's sys.modules would
# prove nothing, since another test may already have imported anything.
_PROBE = """
import importlib, sys

for module in (
    "mtg_rag.embed",
    "mtg_rag.embed.config",
    "mtg_rag.embed.channels",
    "mtg_rag.embed.encoder",
):
    importlib.import_module(module)

heavy = [name for name in ("sentence_transformers", "torch") if name in sys.modules]
assert not heavy, f"importing mtg_rag.embed pulled in: {heavy}"
"""


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


def test_importing_embed_does_not_import_sentence_transformers_or_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


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
