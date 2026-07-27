"""Command-line entry point for the defect sweep.

    just defects
    just defects --format commander

Plans each of the committed themes and prints what the plans got wrong
([ADR 0026]) — repeats, and the prompt's own worked example handed back. The
numbers are a regression signal, never a gate: nothing here fails a run, and
this stays out of `just check` and CI for the reason `just eval` does
([ADR 0020]).

Needs neither corpus nor index. Planning only reads the format template and the
chat model, so this is the cheap sweep — the one to re-run after every change to
`plan/prompt.py`. Scoring curation needs the whole pipeline and is not wired
here yet.
"""

from __future__ import annotations

import argparse
import sys

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.defects.config import DEFAULT_SWEEP_RUNS
from mtg_rag.defects.render import print_plan_sweep
from mtg_rag.defects.sweep import run_plan_sweep
from mtg_rag.templates_config import TEMPLATE_DIR, TEMPLATE_SUFFIX


def _available_formats() -> list[str]:
    return sorted(path.stem for path in TEMPLATE_DIR.glob(f"*{TEMPLATE_SUFFIX}"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.defects",
        description="Plan the committed themes and score what the plans got wrong.",
    )
    parser.add_argument(
        "--format",
        dest="format_name",
        default="commander",
        help="format template to plan against (default: commander)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_SWEEP_RUNS,
        help=(
            "how many times to plan each theme, averaged "
            f"(default: {DEFAULT_SWEEP_RUNS}); 1 is faster but cannot "
            "tell a prompt change from sampling noise"
        ),
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)

    # Checked before the model loads: an unknown format is a cheap, immediate
    # failure rather than one paid for with a weights download.
    available = _available_formats()
    if args.format_name not in available:
        print(
            f"Unknown format {args.format_name!r}. Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return 1

    from mtg_rag.llm import QwenChatClient

    print("Loading the model...")
    try:
        client = QwenChatClient()
    except Exception as error:
        # A missing download, no network, or too little memory all surface here;
        # the user wants one line, not a traceback out of transformers.
        print(f"Could not load the model: {error}", file=sys.stderr)
        return 1

    print(f"Planning the committed themes against {args.format_name}, {args.runs}x each...\n")
    results = run_plan_sweep(format_name=args.format_name, client=client, runs=args.runs)
    print_plan_sweep(results, format_name=args.format_name)

    # Always zero. A sweep that measured a terrible plan still measured it, and
    # a non-zero exit would make this a gate ([ADR 0020]).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
