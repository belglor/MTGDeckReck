"""Command-line entry point for the planner — the whole Plan → Retrieve path.

    just plan "a spooky graveyard deck that mills itself" --colors B
    just plan "elfball ramp into a big finisher" --plan-only

Plans the searches for a free-text theme, then runs them through retrieval and
prints the fused candidate pool. `--plan-only` stops after the queries — the
query-only output this CLI began as, and all it needs is the format template and
the instruct model, not the corpus or the index.

The trap this CLI has to get right: the planner emits only `query_text` /
`purpose`, and the hard constraints (format legality, colour identity, platform)
come from the flags into `Constraints`, **never** from the model ([ADR 0001]).

The full path loads two local models — the planner instruct model and the
embedder — so it plans first and encodes second; the instruct model is sized to
sit alongside the ~1.2 GB embedder on the 8 GB target (`plan/config.py`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.ingest.config import PLATFORMS
from mtg_rag.llm import QwenChatClient
from mtg_rag.plan.planner import plan
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.config import DEFAULT_PLATFORM
from mtg_rag.templates_config import TEMPLATE_DIR, TEMPLATE_SUFFIX

if TYPE_CHECKING:
    import polars as pl

    from mtg_rag.retrieve.filters import Constraints


def _available_formats() -> list[str]:
    return sorted(path.stem for path in TEMPLATE_DIR.glob(f"*{TEMPLATE_SUFFIX}"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.plan",
        description="Plan the searches for a deck request, then retrieve a candidate pool.",
    )
    parser.add_argument("theme", help="a plain-English description of the deck to build")
    parser.add_argument(
        "--format",
        dest="format_name",
        default="commander",
        help="format whose template guides the plan and whose legality filters "
        "(default: commander)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the planned queries and stop, without searching (needs no corpus or index)",
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
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where the corpus and the vector index live (default: data)",
    )
    return parser.parse_args(argv)


def _print_queries(queries: list[PlannedQuery], format_name: str) -> None:
    print(f"\nPlanned {len(queries)} queries for {format_name}:\n")
    for position, query in enumerate(queries, start=1):
        print(f"{position:>3}. {query.query_text}")
        print(f"     {query.purpose}")


def _prepare_retrieval(args: argparse.Namespace) -> tuple[Constraints, pl.DataFrame, Path] | None:
    """The corpus, the vector dir, and the flag-built `Constraints` — or `None`.

    Runs before the planner model loads, so a missing build or an unknown format
    fails cheap. Mirrors the retrieve CLI's existence checks and messages; the
    `Constraints` are assembled from the flags exactly as it does, which is the
    one thing the model must never supply ([ADR 0001]).
    """
    import polars as pl

    from mtg_rag.embed.config import VECTOR_DIR_NAME
    from mtg_rag.ingest.config import CORPUS_NAME
    from mtg_rag.retrieve.filters import Constraints, available_formats, parse_color_identity

    data_dir: Path = args.data_dir
    corpus_path = data_dir / CORPUS_NAME
    vector_dir = data_dir / VECTOR_DIR_NAME

    if not corpus_path.exists():
        print(f"No corpus at {corpus_path}. Run `just ingest` first.", file=sys.stderr)
        return None
    if not vector_dir.exists():
        print(f"No vector index at {vector_dir}. Run `just embed` first.", file=sys.stderr)
        return None

    frame = pl.read_parquet(corpus_path)
    if args.format_name not in available_formats(frame):
        available = ", ".join(sorted(available_formats(frame)))
        print(f"Unknown format {args.format_name!r}. Available: {available}", file=sys.stderr)
        return None

    constraints = Constraints(
        format_name=args.format_name,
        color_identity=parse_color_identity(args.colors),
        platform=args.platform,
    )
    return constraints, frame, vector_dir


def _plan(args: argparse.Namespace) -> list[PlannedQuery] | None:
    """Load the planner, plan the searches, print them — or `None` if it won't load."""
    print("Loading the model...")
    try:
        client = QwenChatClient()
    except Exception as error:
        # A missing weights download, no network, or too little memory all surface
        # here; the user wants one clear line, not a traceback out of transformers.
        print(f"Could not load the planner model: {error}", file=sys.stderr)
        return None

    queries = plan(args.theme, format_name=args.format_name, client=client)
    _print_queries(queries, args.format_name)
    return queries


def _search_and_print(
    queries: list[PlannedQuery],
    *,
    constraints: Constraints,
    frame: pl.DataFrame,
    vector_dir: Path,
) -> None:
    """Run the planned queries through retrieval and print the fused pool.

    Loads the embedder and the store here — after the plan — so `--plan-only`
    never pays for them, and so the two local models load one after the other
    rather than at once.
    """
    import time

    from mtg_rag.embed.encoder import QwenEncoder
    from mtg_rag.retrieve.pool import hydrate, retrieve
    from mtg_rag.retrieve.render import print_pool
    from mtg_rag.store.chroma import open_client

    print("\nLoading the embedder...")
    encoder = QwenEncoder()
    client = open_client(vector_dir)

    started = time.perf_counter()
    pool = retrieve(queries, constraints=constraints, frame=frame, client=client, encoder=encoder)
    elapsed = time.perf_counter() - started

    if not pool:
        print("No candidates. The constraints may be unsatisfiable for these queries.")
        return

    rows = hydrate(frame, [candidate.oracle_id for candidate in pool])
    print(f"\n{len(pool)} candidates in {elapsed * 1000:.0f} ms\n")
    print_pool(pool, rows, explain=False)


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)

    # Validated before the model loads: an unknown format is a cheap, immediate
    # failure, not a reason to spend a weights download first. The choices are the
    # template files themselves — the planner is handed the whole template ([ADR
    # 0023]), so a format exists exactly when its `<format>.md` does.
    available = _available_formats()
    if args.format_name not in available:
        print(
            f"Unknown format {args.format_name!r}. Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return 1

    if args.plan_only:
        return 0 if _plan(args) is not None else 1

    # Full path: check the corpus, index, and format before the planner loads, so
    # a missing build fails without a weights download. Constraints come from the
    # flags here, never from the plan the model returns next ([ADR 0001]).
    prepared = _prepare_retrieval(args)
    if prepared is None:
        return 1
    constraints, frame, vector_dir = prepared

    queries = _plan(args)
    if queries is None:
        return 1

    _search_and_print(queries, constraints=constraints, frame=frame, vector_dir=vector_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
