"""Print a sweep's numbers to the terminal ([ADR 0026]).

Pure display, mirroring `retrieve/render.py` and `curate/render.py`.

Two things it will not do. It prints no verdict — no threshold, no pass/fail,
no colour on a "bad" number — because the instrument reports and never gates
([ADR 0020]), and `role_count` in particular is reported precisely because
[ADR 0006] refuses to say what a right answer would be. And it stamps every
table with what produced it, so a later run is comparable rather than merely
newer (CLAUDE.md's documentation rule).

The thematic / mechanical split gets its own summary rows because that
comparison *is* #72's cross-cutting finding: evocative requests degrade worse
than rules-oriented ones. A single mean would average it away.
"""

from __future__ import annotations

from collections.abc import Sequence

from mtg_rag.defects.scores import RunScores
from mtg_rag.llm_config import MODEL_ID, TEMPERATURE, TOP_K, TOP_P

_THEME_WIDTH = 46


def print_plan_sweep(results: Sequence[RunScores], *, format_name: str) -> None:
    """Print the per-theme plan numbers, the summary rows, and the stamp."""
    print(f"{'theme':<{_THEME_WIDTH}} {'kind':<11} {'queries':>7} {'dup':>7} {'parrot':>7}")
    print("-" * (_THEME_WIDTH + 36))
    for run in results:
        print(f"{_clip(run.theme):<{_THEME_WIDTH}} {run.kind:<11} ", end="")
        if run.plan is None:
            print(f"{'—':>7} {'—':>7} {'—':>7}   PLAN DID NOT VALIDATE")
            continue
        print(
            f"{run.plan.query_count:>7} "
            f"{_rate(run.plan.duplicate_rate):>7} "
            f"{_rate(run.plan.parroting_rate):>7}"
        )

    print("-" * (_THEME_WIDTH + 36))
    _print_summary("all", results)
    for kind in ("thematic", "mechanical"):
        _print_summary(kind, [run for run in results if run.kind == kind])

    failed = sum(run.plan is None for run in results)
    print()
    print(f"model: {MODEL_ID}   format: {format_name}")
    print(f"sampling: temperature {TEMPERATURE}, top-p {TOP_P}, top-k {TOP_K}")
    print(f"themes: {len(results)}   plans that did not validate: {failed}")


def _print_summary(label: str, runs: Sequence[RunScores]) -> None:
    """One mean row over the runs whose plan validated."""
    scored = [run.plan for run in runs if run.plan is not None]
    queries = _mean([float(plan.query_count) for plan in scored])
    duplicate = _mean([plan.duplicate_rate for plan in scored])
    parroting = _mean([plan.parroting_rate for plan in scored])
    print(
        f"{'mean — ' + label:<{_THEME_WIDTH}} {'':<11} "
        f"{_rate(queries, places=1):>7} {_rate(duplicate):>7} {_rate(parroting):>7}"
    )


def _mean(values: Sequence[float | None]) -> float | None:
    """The mean of the defined values, or `None` if none are.

    `None` is undefined, never 0.0 — averaging a rate that was never measured
    into a table would put a false point in it ([ADR 0026], and the rule
    `evals/metrics.py` follows).
    """
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _rate(value: float | None, *, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _clip(theme: str) -> str:
    return theme if len(theme) <= _THEME_WIDTH else theme[: _THEME_WIDTH - 1] + "…"
