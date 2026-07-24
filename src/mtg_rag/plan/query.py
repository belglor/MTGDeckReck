"""The typed query schema the planner emits and retrieval consumes.

Data only — no prompt, no model, no template. [ADR 0004] settles that a plan is
`[{query_text, purpose}]` with the model choosing both the queries and how many,
and that the output is a schema rather than prose *because the app has to
execute it*: a phrasing drift becomes a validation error here instead of a
silently wrong search.

`purpose` is the role a query is covering — "ramp", "theme payoff", whatever the
format template led the model to. Retrieval carries it opaquely: it is attached
to every candidate the query found and displayed, never parsed or checked
against a vocabulary. It exists for curation, which uses it as the starting
hypothesis for a card's role ([ADR 0005]).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    """One search the planner asked for, and what it is meant to cover.

    Both fields are required and non-empty. Whitespace counts as empty: a model
    emitting `"  "` passes a truthiness check while saying nothing, and a plan
    that says nothing should fail at the boundary rather than run.
    """

    query_text: str
    purpose: str

    def __post_init__(self) -> None:
        for field, value in (("query_text", self.query_text), ("purpose", self.purpose)):
            if not value.strip():
                raise ValueError(f"{field} must be a non-empty string, got {value!r}")
