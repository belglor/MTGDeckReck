"""Run the golden set through the real retrieval path and build a report.

Every case goes through `retrieve.pool.retrieve()` — the eval measures the
thing that ships, not a copy of it. What the eval supplies itself is the
`PlannedQuery` list, so a pool is a deterministic function of (corpus, index,
query, constraints); the planner is out of the loop by construction ([ADR 0020]).

**Nothing here fails a run.** A poor number is the output, not an error
([ADR 0011]). The one thing that does raise is a malformed case, and that is
checked up front by `cases.validate_against_corpus` before the model loads.

The report carries a provenance stamp because [ADR 0011]'s baseline-reset rule
is unusable without it: a change to the model, the channel set, or the dimension
opens a new epoch rather than a comparison, and a number with no record of which
geometry produced it cannot be placed in either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
from chromadb.api import ClientAPI

from mtg_rag.embed.config import Channel
from mtg_rag.embed.encoder import Encoder
from mtg_rag.embed.index import VectorIndex
from mtg_rag.evals.cases import EvalCase
from mtg_rag.evals.metrics import base_rate, lift, precision, retention
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.filters import Constraints
from mtg_rag.retrieve.pool import retrieve

#: `purpose` is required on a `PlannedQuery` and carries a role for curation
#: ([ADR 0004]). Nothing downstream of the eval reads it, so one label serves.
EVAL_PURPOSE = "eval case"


@dataclass(frozen=True, slots=True)
class Run:
    """One case under one constraint set."""

    constraints: Constraints
    pool_size: int
    base_rate: float
    precision: float | None
    lift: float | None
    #: Against the case's first constraint set, which is its own reference.
    #: `None` on that first run would be indistinguishable from an undefined
    #: ratio, so it is 1.0 there by construction.
    retention: float | None


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: EvalCase
    runs: tuple[Run, ...]


@dataclass(frozen=True, slots=True)
class Report:
    """A whole eval run, stamped with what produced it.

    The stamp is the index's own sidecar rather than a record of its own: the
    model, dimension and corpus that make two numbers comparable are already
    recorded there by `just embed`, and copying them into a second type would
    be a second place for them to be wrong. Only `k` and the channel set are
    the eval's own ([ADR 0011]).

    Deliberately carries **no aggregate**. A mechanic lift and a theme lift are
    not commensurable, and a mean over them would repeat the mistake [ADR 0008]
    refuses for raw similarity scores.
    """

    index: VectorIndex
    k: int
    channels: tuple[Channel, ...]
    results: tuple[CaseResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": {
                "model_id": self.index.model_id,
                "dim": self.index.dim,
                "corpus_updated_at": self.index.corpus_updated_at,
                "corpus_row_count": self.index.corpus_row_count,
                "k": self.k,
                "channels": list(self.channels),
            },
            "cases": [
                {
                    "id": result.case.id,
                    "query": result.case.query,
                    "predicate": {result.case.predicate.kind: result.case.predicate.value},
                    "rationale": result.case.rationale,
                    "runs": [
                        {
                            "format": run.constraints.format_name,
                            "colors": _colors(run.constraints),
                            "platform": run.constraints.platform,
                            "pool_size": run.pool_size,
                            "base_rate": run.base_rate,
                            "precision": run.precision,
                            "lift": run.lift,
                            "retention": run.retention,
                        }
                        for run in result.runs
                    ],
                }
                for result in self.results
            ],
        }


def _colors(constraints: Constraints) -> str | None:
    """The colour constraint as it would be typed, preserving colorless.

    `None` is unconstrained and `""` is colorless-only — the distinction
    `Constraints` keeps in the type, which the JSON must not flatten.
    """
    if constraints.color_identity is None:
        return None
    return "".join(sorted(constraints.color_identity))


def run_case(
    case: EvalCase,
    *,
    frame: pl.DataFrame,
    client: ClientAPI,
    encoder: Encoder,
    k: int,
    channels: Sequence[Channel],
) -> CaseResult:
    """Every constraint set of one case, in file order.

    The first constraint set is the reference every later run's retention is
    measured against, so order in `golden.toml` is meaningful: put the loosest
    first.
    """
    predicate = case.predicate.expr()
    queries = [PlannedQuery(query_text=case.query, purpose=EVAL_PURPOSE)]

    runs: list[Run] = []
    reference: float | None = None
    for constraints in case.constraints:
        pool = retrieve(
            queries,
            constraints=constraints,
            frame=frame,
            client=client,
            encoder=encoder,
            channels=channels,
            pool_size=k,
        )
        ids = [candidate.oracle_id for candidate in pool]
        base = base_rate(frame, constraints, predicate)
        precision_value = precision(frame, ids, predicate)
        lift_value = lift(precision_value, base)
        if reference is None:
            reference = lift_value
        runs.append(
            Run(
                constraints=constraints,
                pool_size=len(ids),
                base_rate=base,
                precision=precision_value,
                lift=lift_value,
                retention=retention(lift_value, reference),
            )
        )
    return CaseResult(case=case, runs=tuple(runs))


def run_cases(
    cases: Sequence[EvalCase],
    *,
    frame: pl.DataFrame,
    client: ClientAPI,
    encoder: Encoder,
    index: VectorIndex,
    k: int,
    channels: Sequence[Channel],
) -> Report:
    """The whole golden set, in file order.

    One encoder and one client for the run — a case is cheap, and paying the
    model load per case would dominate everything else.
    """
    results = tuple(
        run_case(case, frame=frame, client=client, encoder=encoder, k=k, channels=channels)
        for case in cases
    )
    return Report(index=index, k=k, channels=tuple(channels), results=results)
