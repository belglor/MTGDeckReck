"""Command-line entry point for Scryfall ingestion.

just ingest            # build data/cards.parquet if it is not already current
just ingest --force    # rebuild regardless
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.ingest.config import CORPUS_NAME, DEFAULT_BULK_TYPE, SIDECAR_NAME
from mtg_rag.ingest.normalize import CardRecord, MalformedCardError, build_frame, normalize_card
from mtg_rag.ingest.scryfall import (
    Snapshot,
    download,
    fetch_bulk_entry,
    make_client,
    should_skip,
    stream_cards,
    summarize,
)


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
        default=DEFAULT_BULK_TYPE,
        help=f"Scryfall bulk-data type to ingest (default: {DEFAULT_BULK_TYPE})",
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

            records: list[CardRecord] = []
            skipped: list[str] = []
            for raw in stream_cards(temp_path):
                try:
                    records.append(normalize_card(raw))
                except MalformedCardError as exc:
                    skipped.append(str(exc))

            frame = build_frame(records)
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
    for line in summarize(frame):
        print(f"  {line}")

    if skipped:
        # Surface these rather than swallowing them — a malformed card in
        # Scryfall's own export is worth knowing about.
        print(f"\nSkipped {len(skipped)} malformed card(s):")
        for message in skipped[:5]:
            print(f"  - {message}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
