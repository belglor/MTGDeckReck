"""Tests for Scryfall bulk-index lookup and corpus idempotency.

The card projection itself is covered in `test_ingest_normalize.py`, and the
collapse from printings to cards in `test_ingest_merge.py`. Nothing here
touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mtg_rag.ingest.scryfall import BulkEntry, Snapshot, is_english, should_skip

# --- language --------------------------------------------------------------
# `default_cards` carries a card in a foreign language when it exists in no
# English printing at all — 1,207 Spanish objects, 661 Japanese, 430 French.
# Left in, one of those could win the representative-printing contest and
# supply a card's flavor text, quietly seeding the flavor channel with a
# language the embedding model was never asked about.


def _raw(**overrides: Any) -> dict[str, Any]:
    return {"oracle_id": "abc", "name": "Sol Ring", "lang": "en"} | overrides


def test_english_printings_are_kept() -> None:
    assert is_english(_raw())


def test_non_english_printings_are_skipped() -> None:
    assert not is_english(_raw(lang="es"))
    assert not is_english(_raw(lang="ja"))


def test_a_printing_without_a_language_is_kept() -> None:
    # Scryfall always sets `lang`, so this is defensive — and it defaults to
    # keeping the card, because silently dropping a real card is the worse of
    # the two ways to be wrong.
    raw = _raw()
    del raw["lang"]
    assert is_english(raw)


# --- idempotency -----------------------------------------------------------


def _entry(updated_at: str) -> BulkEntry:
    return BulkEntry(
        bulk_type="oracle_cards",
        updated_at=updated_at,
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        size=1,
    )


def test_skips_when_snapshot_matches_and_corpus_exists(tmp_path: Path) -> None:
    corpus = tmp_path / "cards.parquet"
    corpus.touch()
    snapshot = Snapshot(
        bulk_type="oracle_cards",
        updated_at="2026-07-19T09:03:16.601+00:00",
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        row_count=10,
        ingested_at="2026-07-19T10:00:00+00:00",
    )

    assert should_skip(snapshot, _entry("2026-07-19T09:03:16.601+00:00"), corpus, force=False)


def test_does_not_skip_when_upstream_is_newer(tmp_path: Path) -> None:
    corpus = tmp_path / "cards.parquet"
    corpus.touch()
    snapshot = Snapshot(
        bulk_type="oracle_cards",
        updated_at="2026-07-18T09:00:00.000+00:00",
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        row_count=10,
        ingested_at="2026-07-18T10:00:00+00:00",
    )

    assert not should_skip(snapshot, _entry("2026-07-19T09:03:16.601+00:00"), corpus, force=False)


def test_does_not_skip_when_corpus_is_missing(tmp_path: Path) -> None:
    # Deleting data/ must fully restore on the next run, even though the
    # sidecar still claims the snapshot is current.
    corpus = tmp_path / "cards.parquet"
    snapshot = Snapshot(
        bulk_type="oracle_cards",
        updated_at="2026-07-19T09:03:16.601+00:00",
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        row_count=10,
        ingested_at="2026-07-19T10:00:00+00:00",
    )

    assert not should_skip(snapshot, _entry("2026-07-19T09:03:16.601+00:00"), corpus, force=False)


def test_does_not_skip_without_a_snapshot(tmp_path: Path) -> None:
    corpus = tmp_path / "cards.parquet"
    corpus.touch()
    assert not should_skip(None, _entry("2026-07-19T09:03:16.601+00:00"), corpus, force=False)


def test_force_overrides_a_matching_snapshot(tmp_path: Path) -> None:
    corpus = tmp_path / "cards.parquet"
    corpus.touch()
    snapshot = Snapshot(
        bulk_type="oracle_cards",
        updated_at="2026-07-19T09:03:16.601+00:00",
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        row_count=10,
        ingested_at="2026-07-19T10:00:00+00:00",
    )

    assert not should_skip(snapshot, _entry("2026-07-19T09:03:16.601+00:00"), corpus, force=True)


def test_snapshot_roundtrips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "cards.meta.json"
    snapshot = Snapshot(
        bulk_type="oracle_cards",
        updated_at="2026-07-19T09:03:16.601+00:00",
        download_uri="https://data.scryfall.io/oracle-cards/x.jsonl.gz",
        row_count=31234,
        ingested_at="2026-07-19T10:00:00+00:00",
    )
    snapshot.write(path)

    assert Snapshot.read(path) == snapshot


def test_reading_a_missing_snapshot_returns_none(tmp_path: Path) -> None:
    assert Snapshot.read(tmp_path / "nope.json") is None


def test_reading_a_corrupt_snapshot_returns_none(tmp_path: Path) -> None:
    # A half-written sidecar should force a rebuild, not crash the run.
    path = tmp_path / "cards.meta.json"
    path.write_text("{not json", encoding="utf-8")
    assert Snapshot.read(path) is None
