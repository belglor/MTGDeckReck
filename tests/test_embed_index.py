"""Tests for the vector-index sidecar and its idempotency check.

The sidecar answers exactly one question: is this index current for this parquet
and this model ([ADR 0015])? Everything here is `tmp_path` and pure — no model,
no store, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mtg_rag.embed.config import EMBEDDING_DIM, MODEL_ID
from mtg_rag.embed.index import VectorIndex, should_skip
from mtg_rag.ingest.scryfall import Snapshot

CORPUS_UPDATED_AT = "2026-07-19T09:03:16.601+00:00"


def _snapshot(updated_at: str = CORPUS_UPDATED_AT) -> Snapshot:
    return Snapshot(
        bulk_type="oracle_cards",
        updated_at=updated_at,
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        row_count=38312,
        ingested_at="2026-07-19T10:00:00+00:00",
    )


def _index(
    *,
    model_id: str = MODEL_ID,
    dim: int = EMBEDDING_DIM,
    corpus_updated_at: str = CORPUS_UPDATED_AT,
) -> VectorIndex:
    return VectorIndex(
        model_id=model_id,
        dim=dim,
        corpus_updated_at=corpus_updated_at,
        corpus_row_count=38312,
        channel_counts={"oracle": 33836, "flavor": 20248, "type": 34184},
        embedded_at="2026-07-19T12:00:00+00:00",
    )


@pytest.fixture
def vector_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "vectors"
    directory.mkdir()
    return directory


# --- currency ---------------------------------------------------------------


def test_skips_when_sidecar_matches_corpus_snapshot(vector_dir: Path) -> None:
    assert should_skip(_index(), _snapshot(), vector_dir, force=False)


def test_rebuilds_when_corpus_snapshot_moved(vector_dir: Path) -> None:
    # A refresh that changed no card text still moves the snapshot; the index is
    # rebuilt wholesale rather than diffed (ADR 0015).
    moved = _snapshot("2026-07-20T09:00:00.000+00:00")
    assert not should_skip(_index(), moved, vector_dir, force=False)


def test_rebuilds_when_model_id_changed(vector_dir: Path) -> None:
    # A different model is a different geometry — the vectors are not comparable
    # and the recall baseline resets (ADR 0011).
    stale = _index(model_id="BAAI/bge-base-en-v1.5")
    assert not should_skip(stale, _snapshot(), vector_dir, force=False)


def test_rebuilds_when_dimension_changed(vector_dir: Path) -> None:
    stale = _index(dim=768)
    assert not should_skip(stale, _snapshot(), vector_dir, force=False)


def test_rebuilds_when_vector_dir_is_missing(tmp_path: Path) -> None:
    # Deleting data/ must fully restore on the next run, even though the sidecar
    # still claims the index is current.
    assert not should_skip(_index(), _snapshot(), tmp_path / "gone", force=False)


def test_rebuilds_without_a_sidecar(vector_dir: Path) -> None:
    assert not should_skip(None, _snapshot(), vector_dir, force=False)


def test_rebuilds_without_a_corpus_snapshot(vector_dir: Path) -> None:
    # No corpus provenance means currency cannot be established, so the honest
    # answer is to rebuild rather than assume.
    assert not should_skip(_index(), None, vector_dir, force=False)


def test_force_ignores_a_current_sidecar(vector_dir: Path) -> None:
    assert not should_skip(_index(), _snapshot(), vector_dir, force=True)


# --- persistence ------------------------------------------------------------


def test_index_roundtrips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "vectors.meta.json"
    index = _index()
    index.write(path)

    assert VectorIndex.read(path) == index


def test_reading_a_missing_sidecar_returns_none(tmp_path: Path) -> None:
    assert VectorIndex.read(tmp_path / "nope.json") is None


def test_a_corrupt_sidecar_triggers_a_rebuild(tmp_path: Path, vector_dir: Path) -> None:
    # A half-written sidecar means we cannot trust our own provenance, so read
    # returns None and the run rebuilds rather than crashing or assuming.
    path = tmp_path / "vectors.meta.json"
    path.write_text("{not json", encoding="utf-8")

    assert VectorIndex.read(path) is None
    assert not should_skip(VectorIndex.read(path), _snapshot(), vector_dir, force=False)


def test_channel_counts_survive_a_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "vectors.meta.json"
    _index().write(path)

    restored = VectorIndex.read(path)

    assert restored is not None
    assert restored.channel_counts == {"oracle": 33836, "flavor": 20248, "type": 34184}
