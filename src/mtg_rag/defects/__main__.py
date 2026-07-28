"""Command-line entry point for the defect sweep.

    just defects
    just defects --curate --runs 1

Plans each of the committed themes and prints what the plans got wrong
([ADR 0026]) — repeats, and the prompt's own worked example handed back. The
numbers are a regression signal, never a gate: nothing here fails a run, and
this stays out of `just check` and CI for the reason `just eval` does
([ADR 0020]).

Two sweeps, and the difference in cost is large. The default plans only: it
needs neither corpus nor index, reads just the format template and the chat
model, and is the loop to re-run after every edit to `plan/prompt.py`.
`--curate` runs the whole pipeline and scores the recommendation too, which
means loading the embedder and the index, and paying for curation to decode a
rationale per card it picks — minutes per run rather than seconds.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.defects.config import DEFAULT_SWEEP_RUNS
from mtg_rag.defects.render import print_curation_sweep, print_plan_sweep, print_progress
from mtg_rag.defects.scores import RunScores
from mtg_rag.defects.sweep import run_full_sweep, run_plan_sweep
from mtg_rag.templates_config import TEMPLATE_DIR, TEMPLATE_SUFFIX

#: Where the corpus and index live when `--data-dir` is not given, matching the
#: other CLIs' default.
_DEFAULT_DATA_DIR = Path("data")


def _available_formats() -> list[str]:
    return sorted(path.stem for path in TEMPLATE_DIR.glob(f"*{TEMPLATE_SUFFIX}"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.defects",
        description="Plan the committed themes and score what the output got wrong.",
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
            "how many times to run each theme, averaged "
            f"(default: {DEFAULT_SWEEP_RUNS}); 1 is faster but cannot "
            "tell a change from sampling noise"
        ),
    )
    parser.add_argument(
        "--curate",
        action="store_true",
        help=(
            "also retrieve and curate, and score the recommendation; "
            "needs a built corpus and index, and takes minutes per run"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"where the corpus and index live (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--debug-log",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "append every prompt and raw model reply to PATH as JSONL, including "
            "the ones that failed to validate; the only way to see what a run "
            "that produced nothing actually said"
        ),
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def _load_client(debug_log: Path | None) -> object | None:
    from mtg_rag.llm import QwenChatClient

    print("Loading the model...")
    try:
        client = QwenChatClient()
    except Exception as error:
        # A missing download, no network, or too little memory all surface here;
        # the user wants one line, not a traceback out of transformers.
        print(f"Could not load the model: {error}", file=sys.stderr)
        return None

    if debug_log is None:
        return client

    from mtg_rag.defects.recording import RecordingClient

    print(f"Recording every exchange to {debug_log}")
    return RecordingClient(client, debug_log)


def _run_recorder(client: object) -> Callable[[RunScores], None] | None:
    """A callback writing each finished run into the same log as the exchanges.

    Interleaved rather than dumped at the end, so a sweep that is interrupted —
    or killed after an unexpected hour — still leaves every run it completed.
    Composed with `print_progress` rather than replacing it: the log is opt-in
    ([--debug-log]), progress on the terminal is not.
    """
    from mtg_rag.defects.recording import RecordingClient

    if not isinstance(client, RecordingClient):
        return print_progress

    def record(run: RunScores) -> None:
        print_progress(run)
        client.write({"record": "run"} | asdict(run))

    return record


def _run_plan_only(args: argparse.Namespace) -> list[RunScores] | None:
    client = _load_client(args.debug_log)
    if client is None:
        return None
    print(f"Planning the committed themes against {args.format_name}, {args.runs}x each...\n")
    return run_plan_sweep(
        format_name=args.format_name,
        client=client,  # type: ignore[arg-type]
        runs=args.runs,
        on_run=_run_recorder(client),
    )


def _run_full(args: argparse.Namespace) -> list[RunScores] | None:
    """The whole pipeline. Checks the build before paying for a weights download."""
    import polars as pl

    from mtg_rag.embed.config import VECTOR_DIR_NAME
    from mtg_rag.ingest.config import CORPUS_NAME
    from mtg_rag.retrieve.filters import Constraints, available_formats

    corpus_path = args.data_dir / CORPUS_NAME
    vector_dir = args.data_dir / VECTOR_DIR_NAME
    if not corpus_path.exists():
        print(f"No corpus at {corpus_path}. Run `just ingest` first.", file=sys.stderr)
        return None
    if not vector_dir.exists():
        print(f"No vector index at {vector_dir}. Run `just embed` first.", file=sys.stderr)
        return None

    frame = pl.read_parquet(corpus_path)
    if args.format_name not in available_formats(frame):
        available = ", ".join(sorted(available_formats(frame)))
        print(f"Unknown format {args.format_name!r} in the corpus. Available: {available}")
        return None

    client = _load_client(args.debug_log)
    if client is None:
        return None

    from mtg_rag.embed.encoder import QwenEncoder
    from mtg_rag.store.chroma import open_client

    print("Loading the embedder...")
    encoder = QwenEncoder()
    store = open_client(vector_dir)

    # Constraints are fixed rather than exposed as flags: the numbers compare
    # only within one constraint set, so letting a run vary them would produce a
    # table that cannot be read against any other.
    constraints = Constraints(format_name=args.format_name)

    print(f"Running the committed themes against {args.format_name}, {args.runs}x each...\n")
    return run_full_sweep(
        format_name=args.format_name,
        client=client,  # type: ignore[arg-type]
        frame=frame,
        store=store,
        encoder=encoder,
        constraints=constraints,
        runs=args.runs,
        on_run=_run_recorder(client),
    )


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

    results = _run_full(args) if args.curate else _run_plan_only(args)
    if results is None:
        return 1

    print_plan_sweep(results, format_name=args.format_name)
    if args.curate:
        print()
        print_curation_sweep(results, format_name=args.format_name)

    # Always zero. A sweep that measured terrible output still measured it, and
    # a non-zero exit would make this a gate ([ADR 0020]).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
