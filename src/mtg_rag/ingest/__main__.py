"""Command-line entry point for Scryfall ingestion.

just ingest            # build data/cards.parquet if it is not already current
just ingest --force    # rebuild regardless
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.ingest.config import BULK_TYPE, CORPUS_NAME, SIDECAR_NAME
from mtg_rag.ingest.merge import merge_printings
from mtg_rag.ingest.normalize import CardRecord, MalformedCardError, build_frame, normalize_card
from mtg_rag.ingest.scryfall import (
    Snapshot,
    download,
    fetch_bulk_entry,
    is_english,
    make_client,
    should_skip,
    stream_cards,
    summarize,
)


@dataclass(slots=True)
class ReadStats:
    """What the streaming pass saw, for the CLI to report afterwards.

    `printings` against the final row count is the check that the collapse did
    what it claims — roughly three printings per card.
    """

    printings: int = 0
    skipped: list[str] = field(default_factory=list[str])


def english_records(path: Path, stats: ReadStats) -> Iterator[CardRecord]:
    """Project every English printing in the bulk file, counting as it goes.

    A generator rather than a list: the bulk file holds 116,138 printings, and
    `merge_printings` only ever needs the per-card aggregate, so nothing is
    served by materializing all of them first.
    """
    for raw in stream_cards(path):
        if not is_english(raw):
            continue
        stats.printings += 1
        try:
            yield normalize_card(raw)
        except MalformedCardError as exc:
            stats.skipped.append(str(exc))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.ingest",
        description="Download the Scryfall bulk snapshot into a local parquet corpus.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where the corpus and its provenance sidecar live (default: data)",
    )
    parser.add_argument(
        "--bulk-type",
        default=BULK_TYPE,
        help=f"Scryfall bulk-data type to ingest (default: {BULK_TYPE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the local corpus already matches the upstream snapshot",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)
    data_dir: Path = args.data_dir
    corpus_path = data_dir / CORPUS_NAME
    sidecar_path = data_dir / SIDECAR_NAME

    with make_client() as client:
        print(f"Checking Scryfall bulk index for {args.bulk_type!r}...")
        entry = fetch_bulk_entry(client, args.bulk_type)
        print(f"  upstream snapshot: {entry.updated_at}  ({entry.size / 1e6:.0f} MB)")

        snapshot = Snapshot.read(sidecar_path)
        if should_skip(snapshot, entry, corpus_path, force=args.force):
            print(f"\n{corpus_path} is already current. Nothing to do (use --force to rebuild).")
            return 0

        temp_path = data_dir / ".tmp" / "bulk.jsonl.gz"
        try:
            print(f"  downloading {entry.download_uri}")
            download(client, entry.download_uri, temp_path)

            stats = ReadStats()
            # One card arrives once per printing, so the collapse to one record
            # each ([ADR 0002]) happens here rather than in `build_frame`.
            frame = build_frame(merge_printings(english_records(temp_path, stats)))
        finally:
            temp_path.unlink(missing_ok=True)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(corpus_path)
    Snapshot(
        bulk_type=entry.bulk_type,
        updated_at=entry.updated_at,
        download_uri=entry.download_uri,
        row_count=frame.height,
        ingested_at=datetime.now(UTC).isoformat(),
    ).write(sidecar_path)

    print(f"\nWrote {corpus_path} ({corpus_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  printings:    {stats.printings:,} English  ->  {frame.height:,} cards")
    for line in summarize(frame):
        print(f"  {line}")

    if stats.skipped:
        # Surface these rather than swallowing them — a malformed card in
        # Scryfall's own export is worth knowing about.
        print(f"\nSkipped {len(stats.skipped)} malformed printing(s):")
        for message in stats.skipped[:5]:
            print(f"  - {message}")
        if len(stats.skipped) > 5:
            print(f"  ... and {len(stats.skipped) - 5} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
