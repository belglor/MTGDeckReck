"""Provenance sidecar for the vector index.

Mirrors `ingest.scryfall.Snapshot`: a small JSON file recording what the index
was built from, answering exactly one question — is this index current for this
parquet and this model? If yes, `just embed` is a no-op; if no, every requested
channel is rebuilt from scratch.

There is deliberately no diff and no per-card reconciliation ([ADR 0015]).
Detecting *which* cards changed would need per-channel content hashes stored
somewhere, which [ADR 0010] already declined to add, and acting on the diff
would need three code paths — insert, update, delete — each with its own way for
the index to disagree silently with the parquet. A wholesale rebuild has one
path and cannot drift, because it never carries state forward.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

from mtg_rag.embed.config import EMBEDDING_DIM, MODEL_ID
from mtg_rag.ingest.scryfall import Snapshot


@dataclass(frozen=True, slots=True)
class VectorIndex:
    """What the vector index was built from."""

    model_id: str
    dim: int
    #: Copied from `cards.meta.json`, which ties the index to one corpus build.
    corpus_updated_at: str
    corpus_row_count: int
    channel_counts: Mapping[str, int]
    embedded_at: str

    def write(self, path: Path) -> None:
        payload = {
            "model_id": self.model_id,
            "dim": self.dim,
            "corpus_updated_at": self.corpus_updated_at,
            "corpus_row_count": self.corpus_row_count,
            "channel_counts": dict(self.channel_counts),
            "embedded_at": self.embedded_at,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self | None:
        """Load a sidecar, or return None if it is absent or unusable.

        A corrupt sidecar means we cannot trust our own provenance, so the
        honest response is to rebuild rather than to crash or to assume.
        """
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        data = cast("dict[str, Any]", payload)
        try:
            raw_counts = data["channel_counts"]
            if not isinstance(raw_counts, dict):
                return None
            counts = {
                str(channel): int(count)
                for channel, count in cast("dict[Any, Any]", raw_counts).items()
            }
            return cls(
                model_id=str(data["model_id"]),
                dim=int(data["dim"]),
                corpus_updated_at=str(data["corpus_updated_at"]),
                corpus_row_count=int(data["corpus_row_count"]),
                channel_counts=counts,
                embedded_at=str(data["embedded_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def should_skip(
    index: VectorIndex | None,
    snapshot: Snapshot | None,
    vector_dir: Path,
    *,
    force: bool,
) -> bool:
    """Decide whether the vector index is already current.

    Requires the vector directory to actually exist, not just the sidecar to
    agree — deleting `data/` must fully restore on the next run. A missing
    corpus snapshot means currency cannot be established at all, so the answer
    is to rebuild.
    """
    if force or index is None or snapshot is None:
        return False
    if not vector_dir.exists():
        return False
    return (
        index.model_id == MODEL_ID
        and index.dim == EMBEDDING_DIM
        and index.corpus_updated_at == snapshot.updated_at
    )
