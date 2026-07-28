"""Print a sweep's numbers to the terminal ([ADR 0026]).

Pure display, mirroring `retrieve/render.py` and `curate/render.py`.

Two things it will not do. It prints no verdict — no threshold, no pass/fail,
no colour on a "bad" number — because the instrument reports and never gates
([ADR 0020]), and `role_count` in particular is reported precisely because
[ADR 0006] refuses to say what a right answer would be. And it stamps every
table with what produced it, so a later run is comparable rather than merely
newer (CLAUDE.md's documentation rule).

The sweep repeats each theme, so a theme's row is a mean over its runs. The
**widest spread** observed is printed alongside, and it is the number to read
first: it is the noise floor, and a later change smaller than it has not been
measured, only sampled. Two baseline sweeps with no code change between them
moved the mean duplicate rate from 0.05 to 0.13, which is why this is on the
report rather than in a comment.

The thematic / mechanical split gets its own summary rows because that
comparison *is* #72's cross-cutting finding: evocative requests degrade worse
than rules-oriented ones. A single mean would average it away.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from mtg_rag.defects.scores import PlanScores, RecommendationScores, RunScores
from mtg_rag.llm_config import MODEL_ID, TEMPERATURE, TOP_K, TOP_P

_THEME_WIDTH = 46
_REASON_WIDTH = 90


def print_progress(run: RunScores) -> None:
    """One line the moment a theme finishes.

    `print_plan_sweep` / `print_curation_sweep` only run once, after every
    theme has finished — on a `--curate` sweep (minutes per theme) nothing
    reaches the terminal for the whole run without this. `flush=True` so a
    piped or redirected run still shows it live rather than only once the
    process exits.
    """
    if run.plan is None:
        outcome = "plan did not validate"
    elif run.recommendation is None and run.error is not None:
        outcome = "no recommendation"
    else:
        outcome = "ok"
    seconds = f"{run.seconds:.0f}s" if run.seconds is not None else "?s"
    print(f"  [{seconds:>6}] {run.theme[:60]:<60} {outcome}", flush=True)


def print_plan_sweep(results: Sequence[RunScores], *, format_name: str) -> None:
    """Print the per-theme plan means, the summary rows, the spread, and the stamp."""
    grouped = _group_by_theme(results)

    print(f"{'theme':<{_THEME_WIDTH}} {'kind':<11} {'queries':>7} {'dup':>7} {'parrot':>7}")
    print("-" * (_THEME_WIDTH + 36))
    for theme, runs in grouped.items():
        plans = _validated(runs)
        failed = len(runs) - len(plans)
        note = f"   {failed} of {len(runs)} did not validate" if failed else ""
        print(
            f"{_clip(theme):<{_THEME_WIDTH}} {runs[0].kind:<11} "
            f"{_rate(_mean([float(p.query_count) for p in plans]), places=1):>7} "
            f"{_rate(_mean([p.duplicate_rate for p in plans])):>7} "
            f"{_rate(_mean([p.parroting_rate for p in plans])):>7}{note}"
        )
        _print_reasons(run for run in runs if run.plan is None)

    print("-" * (_THEME_WIDTH + 36))
    _print_summary("all", results)
    for kind in ("thematic", "mechanical"):
        _print_summary(kind, [run for run in results if run.kind == kind])

    _print_stamp(results, grouped, format_name=format_name)


def print_curation_sweep(results: Sequence[RunScores], *, format_name: str) -> None:
    """Print the per-theme recommendation means, the summary rows, and the stamp.

    A table of its own rather than more columns beside the plan's: six metrics
    do not fit on one line, and the two stages fail in different ways, so
    reading them side by side invites a comparison neither supports.
    """
    grouped = _group_by_theme(results)

    print(
        f"{'theme':<{_THEME_WIDTH}} {'kind':<11} {'cards':>6} {'roles':>6} "
        f"{'dup':>6} {'parrot':>7} {'quote':>6} {'type':>6}"
    )
    print("-" * (_THEME_WIDTH + 56))
    for theme, runs in grouped.items():
        scored = _curated(runs)
        missing = len(runs) - len(scored)
        note = f"   {missing} of {len(runs)} produced none" if missing else ""
        head = f"{_clip(theme):<{_THEME_WIDTH}} {runs[0].kind:<11} "
        print(head + _curation_cells(scored) + note)
        _print_reasons(run for run in runs if run.recommendation is None)

    print("-" * (_THEME_WIDTH + 56))
    _print_curation_summary("all", results)
    for kind in ("thematic", "mechanical"):
        _print_curation_summary(kind, [run for run in results if run.kind == kind])

    _print_stamp(results, grouped, format_name=format_name)


def _print_reasons(runs: Iterable[RunScores]) -> None:
    """Why a run produced nothing, indented under its theme's row.

    `_score_one` already records this and the parsers already say something
    specific — "output is not valid JSON", "oracle_id … is not in the retrieved
    pool", "no candidates retrieved". Printing only the count would leave a
    reader unable to tell a model that wrapped prose around its JSON from one
    that invented a card, which are different problems with different fixes.
    """
    for run in runs:
        if run.error:
            print(f"{'':<{_THEME_WIDTH}} {'':<11}   ↳ {_clip_reason(run.error)}")


def _clip_reason(error: str) -> str:
    """One line of it. Parser messages can carry a whole malformed payload."""
    first = error.strip().splitlines()[0] if error.strip() else error
    return first if len(first) <= _REASON_WIDTH else first[: _REASON_WIDTH - 1] + "…"


def _curation_cells(scored: Sequence[RecommendationScores]) -> str:
    return (
        f"{_rate(_mean([float(s.card_count) for s in scored]), places=1):>6} "
        f"{_rate(_mean([_as_float(s.role_count) for s in scored]), places=1):>6} "
        f"{_rate(_mean([s.duplicate_rationale_rate for s in scored])):>6} "
        f"{_rate(_mean([s.parroting_rate for s in scored])):>7} "
        f"{_rate(_mean([s.self_quotation_rate for s in scored])):>6} "
        f"{_rate(_mean([s.false_type_claim_rate for s in scored])):>6}"
    )


def _print_curation_summary(label: str, runs: Sequence[RunScores]) -> None:
    scored = _curated(runs)
    print(f"{'mean — ' + label:<{_THEME_WIDTH}} {'':<11} " + _curation_cells(scored))


def _curated(runs: Sequence[RunScores]) -> list[RecommendationScores]:
    return [run.recommendation for run in runs if run.recommendation is not None]


def _as_float(value: int | None) -> float | None:
    return None if value is None else float(value)


def _print_summary(label: str, runs: Sequence[RunScores]) -> None:
    """One mean row over every validated run, across themes."""
    plans = _validated(runs)
    print(
        f"{'mean — ' + label:<{_THEME_WIDTH}} {'':<11} "
        f"{_rate(_mean([float(p.query_count) for p in plans]), places=1):>7} "
        f"{_rate(_mean([p.duplicate_rate for p in plans])):>7} "
        f"{_rate(_mean([p.parroting_rate for p in plans])):>7}"
    )


def _print_stamp(
    results: Sequence[RunScores],
    grouped: dict[str, list[RunScores]],
    *,
    format_name: str,
) -> None:
    """What produced the table, and how much of it is noise."""
    per_theme = {len(runs) for runs in grouped.values()}
    runs_note = str(per_theme.pop()) if len(per_theme) == 1 else "varies"
    failed = sum(run.plan is None for run in results)

    print()
    print(f"model: {MODEL_ID}   format: {format_name}")
    print(f"sampling: temperature {TEMPERATURE}, top-p {TOP_P}, top-k {TOP_K}")
    print(f"themes: {len(grouped)}   runs per theme: {runs_note}   did not validate: {failed}")

    dup = _widest(grouped, lambda plan: plan.duplicate_rate)
    parrot = _widest(grouped, lambda plan: plan.parroting_rate)
    print(f"widest spread within one theme: dup {_rate(dup)}, parrot {_rate(parrot)}")
    print("a later change smaller than that spread has not been measured, only sampled")


def _widest(
    grouped: dict[str, list[RunScores]],
    read: Callable[[PlanScores], float | None],
) -> float | None:
    """The largest max-minus-min any single theme showed for one metric."""
    spreads: list[float] = []
    for runs in grouped.values():
        values = [value for plan in _validated(runs) if (value := read(plan)) is not None]
        if len(values) > 1:
            spreads.append(max(values) - min(values))
    return max(spreads) if spreads else None


def _group_by_theme(results: Sequence[RunScores]) -> dict[str, list[RunScores]]:
    """Runs keyed by theme, in first-appearance order so the table is stable."""
    grouped: dict[str, list[RunScores]] = {}
    for run in results:
        grouped.setdefault(run.theme, []).append(run)
    return grouped


def _validated(runs: Sequence[RunScores]) -> list[PlanScores]:
    return [run.plan for run in runs if run.plan is not None]


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
