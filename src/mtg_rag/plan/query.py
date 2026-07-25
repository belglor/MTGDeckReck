"""The typed query schema the planner emits and retrieval consumes.

Data only — no prompt, no model, no template. [ADR 0004] settles that a plan is
`[{query_text, purpose}]`, the model choosing the queries and how many, as a
schema rather than prose *because the app executes it*: a phrasing drift becomes
a validation error here, not a silently wrong search.

`purpose` is the role a query covers — "ramp", "theme payoff". Retrieval carries
it opaquely, attaching it to each candidate the query found but never parsing it;
it exists for curation, as the starting hypothesis for a card's role ([ADR 0005]).
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
