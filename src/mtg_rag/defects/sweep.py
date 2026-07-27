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

Two entry points, and the cost between them is not close. `run_plan_sweep` needs
no corpus, no index and no embedder — only the chat model — so it is the loop
Phase 3's grounding work iterates on. `run_full_sweep` runs the whole pipeline
and pays for curation to decode a rationale per card it picks, which is minutes
per run rather than seconds.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from chromadb.api import ClientAPI

from mtg_rag.curate.cards import curation_cards
from mtg_rag.curate.config import CURATION_POOL_SIZE
from mtg_rag.curate.curation import curate
from mtg_rag.curate.parse import MalformedRecommendationError
from mtg_rag.defects.config import DEFAULT_SWEEP_RUNS, SWEEP_THEMES
from mtg_rag.defects.scores import RunScores, score_plan, score_recommendation
from mtg_rag.embed.encoder import Encoder
from mtg_rag.llm import LLMClient
from mtg_rag.plan.parse import MalformedPlanError
from mtg_rag.plan.planner import plan
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.retrieve.pool import hydrate, retrieve


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


def run_full_sweep(
    *,
    format_name: str,
    client: LLMClient,
    frame: pl.DataFrame,
    store: ClientAPI,
    encoder: Encoder,
    constraints: Constraints,
    themes: Sequence[tuple[str, str]] = SWEEP_THEMES,
    runs: int = DEFAULT_SWEEP_RUNS,
) -> list[RunScores]:
    """Plan, retrieve and curate each theme `runs` times, scoring both stages.

    The expensive sweep. Curation decodes a rationale per card it picks, so a
    run costs minutes rather than seconds — `run_plan_sweep` is the loop to use
    when only the plan matters.

    `constraints` is fixed by the caller rather than varied per theme: the
    numbers are comparable only within one constraint set, so a sweep that
    changed them mid-run would produce a table that cannot be read.

    Every stage failure is recorded rather than raised, for the reason
    `run_plan_sweep` gives — an instrument that stops at the first bad output
    cannot measure how often output is bad ([ADR 0020]). A run that fails to
    plan has no pool to curate and reports neither score; one that plans but
    retrieves nothing, or whose recommendation never validates, keeps its plan
    score and leaves the recommendation undefined.
    """
    results: list[RunScores] = []
    for theme, kind in themes:
        for _ in range(runs):
            results.append(
                _score_one(
                    theme,
                    kind,
                    format_name=format_name,
                    client=client,
                    frame=frame,
                    store=store,
                    encoder=encoder,
                    constraints=constraints,
                )
            )
    return results


def _score_one(
    theme: str,
    kind: str,
    *,
    format_name: str,
    client: LLMClient,
    frame: pl.DataFrame,
    store: ClientAPI,
    encoder: Encoder,
    constraints: Constraints,
) -> RunScores:
    """One theme, once, through the whole pipeline."""
    try:
        queries = plan(theme, format_name=format_name, client=client)
    except MalformedPlanError as error:
        return RunScores(theme=theme, kind=kind, plan=None, error=str(error))

    plan_scores = score_plan([query.query_text for query in queries])
    pool = retrieve(queries, constraints=constraints, frame=frame, client=store, encoder=encoder)
    if not pool:
        # A real answer, not a failure: the constraints may be unsatisfiable for
        # these queries. There is nothing to curate, so nothing to score.
        return RunScores(theme=theme, kind=kind, plan=plan_scores, error="no candidates retrieved")

    rows = hydrate(frame, [candidate.oracle_id for candidate in pool])
    # Curation sees the top of the pool, not all of it ([#93] measured the cap).
    # Cut on the rows rather than the pool, as `just plan` does, so every row
    # still has a candidate behind it to read purposes from.
    shown = rows.head(CURATION_POOL_SIZE)
    try:
        recommendation = curate(
            theme, format_name=format_name, cards=curation_cards(pool, shown), client=client
        )
    except MalformedRecommendationError as error:
        return RunScores(theme=theme, kind=kind, plan=plan_scores, error=str(error))

    # Scored against `shown`, not the whole pool: those are the cards curation
    # was given, and its ids are closed to that set ([ADR 0024]).
    return RunScores(
        theme=theme,
        kind=kind,
        plan=plan_scores,
        recommendation=score_recommendation(recommendation, shown),
    )
