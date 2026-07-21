"""Embedding configuration.

Constants only. The channel vocabulary lives here rather than beside the
composition logic so `channels.py` holds behavior and nothing else, and so the
store and the CLI can name a channel without importing the composition module.

The three channels are settled in [ADR 0007]: oracle text (with the card name
folded in), flavor text, and type line. They are embedded separately because
each is a different register of language — terse rules prose, evocative
narrative, and a controlled vocabulary — and averaging them into one vector
lets the longest dominate and blurs the signal that makes each useful.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

#: One semantic channel. Doubles as the per-channel collection suffix.
type Channel = Literal["oracle", "flavor", "type"]

#: Which corpus columns each channel embeds, in join order — the channel-to-
#: content pairing. Kept here so re-pointing a channel, or adding one, is an
#: edit to data rather than to the composition logic that reads it.
#:
#: The card name appears under `oracle` and nowhere else, and that is a decision
#: rather than an omission ([ADR 0007]): a name is a rules-text-adjacent
#: identifier, so it belongs with rules prose, while the type line is a
#: controlled vocabulary that a name would dilute. Adding "name" to another
#: channel changes retrieval behavior and wants the ADR revisited, not just this
#: mapping edited.
CHANNEL_SOURCES: Mapping[Channel, tuple[str, ...]] = {
    "oracle": ("name", "oracle_text"),
    "flavor": ("flavor_text",),
    "type": ("type_line",),
}

#: Joins a channel's columns where it reads more than one. Only `oracle` does
#: today, so this is deliberately one separator rather than one per channel.
CHANNEL_SOURCE_SEPARATOR = "\n"

#: Every channel, in the order they are built and reported. Derived, so the
#: mapping above stays the single place a channel is added or removed.
CHANNELS: tuple[Channel, ...] = tuple(CHANNEL_SOURCES)

#: Name of the composed-text column in the frame `channel_frame` returns. The
#: encoder and the CLI read it, so it is named once here rather than spelled
#: out at each call site.
TEXT_COLUMN = "text"

#: On-disk names for the vector index and its provenance sidecar, under whatever
#: --data-dir the CLI is given. Mirrors `ingest/config.py`'s corpus names.
VECTOR_DIR_NAME = "vectors"
VECTOR_SIDECAR_NAME = "vectors.meta.json"

#: The embedding model ([ADR 0012]): Apache 2.0, ~600M parameters, and the
#: strongest open-weight option that fits the target RTX 2070.
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

#: Vector width the model emits. Changing this invalidates the index and resets
#: the recall baseline ([ADR 0011]) — the sidecar compares it for that reason.
#: The model is trained with Matryoshka representation learning, so this is
#: truncatable later without retraining, as a measured change rather than a
#: guess.
EMBEDDING_DIM = 1024

#: Token cap per text. The model's own default is 32K, which would pad every
#: short card to an absurd width; oracle text runs to a p99 of ~448 characters,
#: so this bounds padding cost without truncating real cards.
MAX_SEQ_LENGTH = 512

#: Texts per forward pass. Documents are embedded in bulk during `just embed`;
#: queries arrive a few at a time per request, so they use a smaller batch.
DOCUMENT_BATCH_SIZE = 128
QUERY_BATCH_SIZE = 32

#: Attention kernel. `sdpa` ships with torch, needs no extra package, and runs
#: on every backend below. flash-attention-2 is deliberately not requested: it
#: would need `flash-attn` installed and Ampere or newer, and it is not a
#: dependency this project carries.
ATTENTION_IMPLEMENTATION = "sdpa"

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
