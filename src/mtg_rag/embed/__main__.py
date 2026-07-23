"""Command-line entry point for the embedding pass.

just embed                    # build data/vectors/ if it is not already current
just embed --force            # rebuild regardless
just embed --channel flavor   # rebuild one channel, for iterating locally

The first run downloads ~1.2 GB of model weights.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from chromadb.api import ClientAPI

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.embed.channels import channel_frame
from mtg_rag.embed.config import (
    CHANNELS,
    DOCUMENT_BATCH_SIZE,
    EMBEDDING_DIM,
    MODEL_ID,
    TEXT_COLUMN,
    VECTOR_DIR_NAME,
    VECTOR_SIDECAR_NAME,
    Channel,
)
from mtg_rag.embed.encoder import Encoder, QwenEncoder
from mtg_rag.embed.index import VectorIndex, should_skip
from mtg_rag.ingest.config import CORPUS_NAME, SIDECAR_NAME
from mtg_rag.ingest.scryfall import Snapshot
from mtg_rag.store.chroma import open_client, reset_collection, write_vectors


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.embed",
        description="Embed the card corpus into one Chroma collection per channel.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where the corpus and the vector index live (default: data)",
    )
    parser.add_argument(
        "--channel",
        action="append",
        choices=CHANNELS,
        dest="channels",
        help="channel to rebuild; repeatable (default: all three)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DOCUMENT_BATCH_SIZE,
        help=f"texts per forward pass (default: {DOCUMENT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the index already matches the corpus and model",
    )
    return parser.parse_args(argv)


def embed_channel(
    encoder: Encoder,
    client: ClientAPI,
    frame: pl.DataFrame,
    channel: Channel,
    *,
    batch_size: int,
) -> int:
    """Rebuild one channel's collection from the corpus. Returns vectors written."""
    composed = channel_frame(frame, channel)
    ids: list[str] = composed[ID_COLUMN].to_list()
    texts: list[str] = composed[TEXT_COLUMN].to_list()

    vectors = encoder.encode_documents(texts, batch_size=batch_size)

    collection = reset_collection(client, channel)
    # Iterating a 2-D array yields row views, so this pairs ids to vectors
    # without copying the embeddings.
    return write_vectors(client, collection, dict(zip(ids, vectors, strict=True)))


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)
    data_dir: Path = args.data_dir
    corpus_path = data_dir / CORPUS_NAME
    vector_dir = data_dir / VECTOR_DIR_NAME
    vector_sidecar = data_dir / VECTOR_SIDECAR_NAME

    if not corpus_path.exists():
        print(f"No corpus at {corpus_path}. Run `just ingest` first.", file=sys.stderr)
        return 1

    snapshot = Snapshot.read(data_dir / SIDECAR_NAME)
    index = VectorIndex.read(vector_sidecar)
    if should_skip(index, snapshot, vector_dir, force=args.force):
        print(f"{vector_dir} is already current. Nothing to do (use --force to rebuild).")
        return 0

    requested: tuple[Channel, ...] = tuple(args.channels) if args.channels else CHANNELS
    frame = pl.read_parquet(corpus_path)
    print(f"Corpus:  {corpus_path} ({frame.height:,} rows)")
    print(f"Model:   {MODEL_ID} ({EMBEDDING_DIM}-dim)")
    print(f"Channels: {', '.join(requested)}\n")

    print("Loading the model (first run downloads ~1.2 GB)...")
    encoder = QwenEncoder()
    client = open_client(vector_dir)

    counts: dict[str, int] = {}
    started = time.perf_counter()
    for channel in requested:
        channel_started = time.perf_counter()
        counts[channel] = embed_channel(encoder, client, frame, channel, batch_size=args.batch_size)
        elapsed = time.perf_counter() - channel_started
        print(f"  {channel:7s} {counts[channel]:>7,} vectors  ({elapsed / 60:.1f} min)")
    total_elapsed = time.perf_counter() - started

    total = sum(counts.values())
    print(f"\nWrote {total:,} vectors to {vector_dir} in {total_elapsed / 60:.1f} min")

    if set(requested) == set(CHANNELS):
        VectorIndex(
            model_id=MODEL_ID,
            dim=EMBEDDING_DIM,
            corpus_updated_at=snapshot.updated_at if snapshot else "",
            corpus_row_count=frame.height,
            channel_counts=counts,
            embedded_at=datetime.now(UTC).isoformat(),
        ).write(vector_sidecar)
        print(f"Recorded {vector_sidecar}")
    else:
        # The sidecar states that the whole index is current, which a partial
        # run cannot establish — leaving it alone keeps the next run honest.
        print("Partial rebuild, so the sidecar was left unchanged.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
