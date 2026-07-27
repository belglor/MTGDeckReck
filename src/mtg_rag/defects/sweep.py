"""Drive the themes through a stage and score what comes back ([ADR 0026]).

The I/O half of the instrument: `scores.py` does the arithmetic, this calls the
model. Kept apart so the arithmetic is testable without a multi-GB download and
this stays thin enough to read.

**A malformed plan is recorded, not raised.** The planner itself is
validate-or-raise ([ADR 0022]) and stays that way — but an instrument that dies
on the first bad plan cannot measure how often plans are bad, which is the
question #72 asks. [ADR 0020]'s rule that the instrument reports and never fails
a run matters most for the runs that went worst, so the exception is caught here
and becomes a row in the table.

Planning needs no corpus, no index and no embedder — only the chat model — so
this sweep is the cheap one, and the path Phase 3's grounding work iterates on.
Scoring curation needs the whole pipeline and is a separate entry point.
"""

from __future__ import annotations

from collections.abc import Sequence

from mtg_rag.defects.config import DEFAULT_SWEEP_RUNS, SWEEP_THEMES
from mtg_rag.defects.scores import RunScores, score_plan
from mtg_rag.llm import LLMClient
from mtg_rag.plan.parse import MalformedPlanError
from mtg_rag.plan.planner import plan


def run_plan_sweep(
    *,
    format_name: str,
    client: LLMClient,
    themes: Sequence[tuple[str, str]] = SWEEP_THEMES,
    runs: int = DEFAULT_SWEEP_RUNS,
) -> list[RunScores]:
    """Plan each theme `runs` times and score the queries, one `RunScores` each.

    Every repeat is returned rather than averaged here, so the caller keeps the
    spread. Sampling is stochastic and the noise is large — two baseline sweeps
    with no code change between them moved the mean duplicate rate 0.05 to 0.13
    (`DEFAULT_SWEEP_RUNS` has the numbers) — so a report that averaged in here
    would hide the one thing a reader needs to judge whether a change is real.

    `themes` defaults to the committed set; passing another starts a new
    baseline rather than continuing this one, so the caller that does it owns
    saying so.

    A run whose plan never validates gets a row with `plan=None` and the failure
    in `error`, and the sweep continues. Its siblings for the same theme are
    unaffected.
    """
    results: list[RunScores] = []
    for theme, kind in themes:
        for _ in range(runs):
            try:
                queries = plan(theme, format_name=format_name, client=client)
            except MalformedPlanError as error:
                results.append(RunScores(theme=theme, kind=kind, plan=None, error=str(error)))
                continue
            results.append(
                RunScores(theme=theme, kind=kind, plan=score_plan([q.query_text for q in queries]))
            )
    return results
