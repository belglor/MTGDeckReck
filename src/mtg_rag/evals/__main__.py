"""Command-line entry point for the eval.

    just eval
    just eval --case graveyard --case madness
    just eval -k 50

Prints one flat table and writes a JSON report. There is no aggregate and no
pass/fail: a mechanic lift and a theme lift are not commensurable, and the
numbers are a regression signal rather than a gate ([ADR 0011], [ADR 0020]).
The process exits non-zero only when it could not run — a missing corpus, a
missing index, or a malformed case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.embed.config import CHANNELS, VECTOR_DIR_NAME, VECTOR_SIDECAR_NAME
from mtg_rag.embed.encoder import QwenEncoder
from mtg_rag.embed.index import VectorIndex
from mtg_rag.evals.cases import (
    MalformedCaseError,
    golden_path,
    load_cases,
    validate_against_corpus,
)
from mtg_rag.evals.config import DEFAULT_EVAL_K, REPORT_NAME
from mtg_rag.evals.runner import Report, run_cases
from mtg_rag.ingest.config import CORPUS_NAME
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.store.chroma import open_client


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.evals",
        description="Run the golden set and report retrieval lift.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="case id to run; repeatable (default: every case)",
    )
    parser.add_argument(
        "-k",
        "--pool-size",
        type=int,
        default=DEFAULT_EVAL_K,
        help=f"pool depth every case is measured at (default: {DEFAULT_EVAL_K})",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="an alternative golden set (default: the shipped one)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where the corpus and the vector index live (default: data)",
    )
    return parser.parse_args(argv)


def _format(value: float | None, spec: str) -> str:
    """A missing number reads as `--`, never as zero."""
    return "--" if value is None else format(value, spec)


def _colors_label(constraints: Constraints) -> str:
    """The colour constraint for display, keeping colorless distinct from any."""
    colors = constraints.color_identity
    if colors is None:
        return "any"
    return "".join(sorted(colors)) if colors else "colorless"


def _print_report(report: Report) -> None:
    index = report.index
    print(f"Model:    {index.model_id} (dim {index.dim})")
    print(f"Corpus:   {index.corpus_row_count:,} rows, updated {index.corpus_updated_at}")
    print(f"Channels: {', '.join(report.channels)}")
    print(f"k:        {report.k}\n")

    header = (
        f"{'case':<12} {'constraints':<22} {'pool':>5} "
        f"{'base':>8} {'prec':>7} {'lift':>9} {'ret':>6}"
    )
    print(header)
    print("-" * len(header))
    for result in report.results:
        for position, run in enumerate(result.runs):
            where = f"{run.constraints.format_name}/{_colors_label(run.constraints)}"
            print(
                f"{result.case.id if position == 0 else '':<12} "
                f"{where:<22} "
                f"{run.pool_size:>5} "
                f"{_format(run.base_rate, '.2%'):>8} "
                f"{_format(run.precision, '.1%'):>7} "
                f"{_format(run.lift, '.1f') + 'x':>9} "
                f"{_format(run.retention, '.2f'):>6}"
            )


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)
    data_dir: Path = args.data_dir
    corpus_path = data_dir / CORPUS_NAME
    vector_dir = data_dir / VECTOR_DIR_NAME
    sidecar_path = data_dir / VECTOR_SIDECAR_NAME

    if not corpus_path.exists():
        print(f"No corpus at {corpus_path}. Run `just ingest` first.", file=sys.stderr)
        return 1
    if not vector_dir.exists():
        print(f"No vector index at {vector_dir}. Run `just embed` first.", file=sys.stderr)
        return 1

    index = VectorIndex.read(sidecar_path)
    if index is None:
        # Without the stamp a number cannot be placed against any baseline
        # ([ADR 0011]), so running would produce a figure nobody may compare.
        print(
            f"No usable index sidecar at {sidecar_path}. Run `just embed` first.",
            file=sys.stderr,
        )
        return 1

    try:
        cases = load_cases(args.golden or golden_path())
    except MalformedCaseError as error:
        print(f"Malformed golden set: {error}", file=sys.stderr)
        return 1

    if args.case_ids:
        wanted = set(args.case_ids)
        unknown = wanted - {case.id for case in cases}
        if unknown:
            print(f"Unknown case id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        cases = tuple(case for case in cases if case.id in wanted)

    frame = pl.read_parquet(corpus_path)
    try:
        # Before the model loads: a mistyped keyword should cost milliseconds.
        validate_against_corpus(cases, frame)
    except MalformedCaseError as error:
        print(f"Malformed golden set: {error}", file=sys.stderr)
        return 1

    print("Loading the model...\n")
    encoder = QwenEncoder()
    client = open_client(vector_dir)

    started = time.perf_counter()
    report = run_cases(
        cases,
        frame=frame,
        client=client,
        encoder=encoder,
        index=index,
        k=args.pool_size,
        channels=CHANNELS,
    )
    elapsed = time.perf_counter() - started

    _print_report(report)

    report_path = data_dir / REPORT_NAME
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    cases_run = len(report.results)
    runs = sum(len(result.runs) for result in report.results)
    print(
        f"\n{cases_run} case{'' if cases_run == 1 else 's'}, "
        f"{runs} run{'' if runs == 1 else 's'} in {elapsed:.1f}s -> {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
