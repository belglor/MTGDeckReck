"""Command-line entry point for retrieval.

just retrieve "graveyard recursion" "self-mill enablers" --format commander --colors B
just retrieve "mana rocks" --colors WU --explain
just retrieve "goblins" --channel flavor      # one channel, to see what it contributes

Exists so the retrieval path is exercisable before a planner does. Each
positional argument becomes one query; `--purpose` labels them all, standing in
for what the planner will eventually choose per query.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.embed.config import CHANNELS, VECTOR_DIR_NAME, Channel
from mtg_rag.embed.encoder import QwenEncoder
from mtg_rag.ingest.config import CORPUS_NAME, PLATFORMS
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.config import CHANNEL_TOP_K, DEFAULT_PLATFORM, DEFAULT_POOL_SIZE
from mtg_rag.retrieve.filters import Constraints, available_formats
from mtg_rag.retrieve.fusion import Candidate
from mtg_rag.retrieve.pool import hydrate, retrieve
from mtg_rag.store.chroma import open_client

#: What `--purpose` says when the caller does not. A planner supplies a real
#: role per query ([ADR 0004]); typing one by hand every run is noise.
CLI_PURPOSE = "cli query"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.retrieve",
        description="Search the card index and print a fused candidate pool.",
    )
    parser.add_argument("queries", nargs="+", help="one or more search phrases")
    parser.add_argument(
        "--format",
        dest="format_name",
        default="commander",
        help="format whose legality applies (default: commander)",
    )
    parser.add_argument(
        "--colors",
        default=None,
        help="color identity to stay within, e.g. BG. Omit for no color "
        "constraint; pass '' for colorless-only",
    )
    parser.add_argument(
        "--platform",
        default=DEFAULT_PLATFORM,
        choices=PLATFORMS,
        help=f"where the cards must be playable (default: {DEFAULT_PLATFORM})",
    )
    parser.add_argument(
        "--channel",
        action="append",
        choices=CHANNELS,
        dest="channels",
        help="channel to search; repeatable (default: all three)",
    )
    parser.add_argument(
        "-k",
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=f"how many candidates to print (default: {DEFAULT_POOL_SIZE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=CHANNEL_TOP_K,
        help=f"how deep each channel ranking goes (default: {CHANNEL_TOP_K})",
    )
    parser.add_argument(
        "--purpose",
        default=CLI_PURPOSE,
        help=f"role label attached to every query (default: {CLI_PURPOSE!r})",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="show which query, channel and rank found each candidate",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where the corpus and the vector index live (default: data)",
    )
    return parser.parse_args(argv)


def _color_identity(colors: str | None) -> frozenset[str] | None:
    """`--colors` as a constraint.

    Absent means unconstrained; an empty string means colorless-only. Those are
    genuinely different requests, so the flag distinguishes them.
    """
    if colors is None:
        return None
    return frozenset(colors.strip().upper())


def _print_pool(pool: list[Candidate], rows: pl.DataFrame, *, explain: bool) -> None:
    by_id = {row["oracle_id"]: row for row in rows.iter_rows(named=True)}
    for position, candidate in enumerate(pool, start=1):
        card = by_id.get(candidate.oracle_id)
        if card is None:  # pragma: no cover - hydration drops unknown ids
            continue
        cost = card["mana_cost"] or ""
        print(f"{position:>3}. {card['name']}  {cost}")
        print(f"     {card['type_line']}   (score {candidate.score:.4f})")
        if explain:
            for source in candidate.sources:
                # Distance is shown, never ranked on: it says whether this
                # channel was confident, which rank alone cannot ([ADR 0008]).
                print(
                    f"       - {source.channel:<7} rank {source.rank:<3} "
                    f"dist {source.distance:.3f}  {source.purpose}"
                )


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)
    data_dir: Path = args.data_dir
    corpus_path = data_dir / CORPUS_NAME
    vector_dir = data_dir / VECTOR_DIR_NAME

    if not corpus_path.exists():
        print(f"No corpus at {corpus_path}. Run `just ingest` first.", file=sys.stderr)
        return 1
    if not vector_dir.exists():
        print(f"No vector index at {vector_dir}. Run `just embed` first.", file=sys.stderr)
        return 1

    frame = pl.read_parquet(corpus_path)
    if args.format_name not in available_formats(frame):
        available = ", ".join(sorted(available_formats(frame)))
        print(f"Unknown format {args.format_name!r}. Available: {available}", file=sys.stderr)
        return 1

    constraints = Constraints(
        format_name=args.format_name,
        color_identity=_color_identity(args.colors),
        platform=args.platform,
    )
    channels: tuple[Channel, ...] = tuple(args.channels) if args.channels else CHANNELS
    queries = [PlannedQuery(query_text=text, purpose=args.purpose) for text in args.queries]

    print(f"Corpus:   {corpus_path} ({frame.height:,} rows)")
    print(f"Filter:   {args.format_name}, colors={args.colors or 'any'}, {args.platform}")
    print(f"Channels: {', '.join(channels)}")
    print(f"Queries:  {len(queries)}\n")

    print("Loading the model...")
    encoder = QwenEncoder()
    client = open_client(vector_dir)

    started = time.perf_counter()
    pool = retrieve(
        queries,
        constraints=constraints,
        frame=frame,
        client=client,
        encoder=encoder,
        channels=channels,
        top_k=args.top_k,
        pool_size=args.pool_size,
    )
    elapsed = time.perf_counter() - started

    if not pool:
        print("No candidates. The constraints may be unsatisfiable for these queries.")
        return 0

    rows = hydrate(frame, [candidate.oracle_id for candidate in pool])
    print(f"{len(pool)} candidates in {elapsed * 1000:.0f} ms\n")
    _print_pool(pool, rows, explain=args.explain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
