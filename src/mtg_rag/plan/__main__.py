"""Command-line entry point for the planner.

    just plan "a spooky graveyard deck that mills itself"
    just plan "elfball ramp into a big finisher" --format commander

Runs the Plan stage alone: a free-text theme in, the validated
`list[PlannedQuery]` out, each with its purpose. Exists so the planner is
exercisable before it is wired into retrieval end-to-end, the way `just retrieve`
stands in for the searches a planner will eventually drive.

It needs the format template and the local instruct model — but not the corpus or
the vector index, which the planner never reads. So, unlike the retrieve CLI,
there are no `just ingest` / `just embed` existence checks here.
"""

from __future__ import annotations

import argparse
import sys

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.plan.client import QwenPlannerClient
from mtg_rag.plan.config import TEMPLATE_DIR, TEMPLATE_SUFFIX
from mtg_rag.plan.planner import plan


def _available_formats() -> list[str]:
    return sorted(path.stem for path in TEMPLATE_DIR.glob(f"*{TEMPLATE_SUFFIX}"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.plan",
        description="Plan the searches for a deck request and print them.",
    )
    parser.add_argument("theme", help="a plain-English description of the deck to build")
    parser.add_argument(
        "--format",
        dest="format_name",
        default="commander",
        help="format whose template guides the plan (default: commander)",
    )
    return parser.parse_args(argv)


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

    print("Loading the model...")
    try:
        client = QwenPlannerClient()
    except Exception as error:
        # A missing weights download, no network, or too little memory all surface
        # here; the user wants one clear line, not a traceback out of transformers.
        print(f"Could not load the planner model: {error}", file=sys.stderr)
        return 1

    queries = plan(args.theme, format_name=args.format_name, client=client)

    print(f"\nPlanned {len(queries)} queries for {args.format_name}:\n")
    for position, query in enumerate(queries, start=1):
        print(f"{position:>3}. {query.query_text}")
        print(f"     {query.purpose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
