"""Turn a deck theme into the set of retrieval queries we actually run.

The planner decides *what* to look for; this module decides how those lookups
are shaped and which candidates survive the hard constraints afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mtg_rag.retrieve.constraints import CompletionClient, Constraints, infer_constraints

# Roles a Commander deck needs covered regardless of theme. The format template
# owns the real list; this is the fallback when the planner returns nothing.
_FALLBACK_ROLES = ("theme payoff", "enabler", "ramp", "card draw", "removal")


@dataclass(frozen=True)
class PlannedQuery:
    """One retrieval pass: what to search for, and why we wanted it."""

    query_text: str
    purpose: str


@dataclass(frozen=True)
class Card:
    """The slice of a Scryfall record the orchestrator cares about."""

    name: str
    colors: tuple[str, ...]
    legalities: dict[str, str]


class QueryOrchestrator:
    """Plans the queries for a theme and narrows what comes back."""

    def __init__(self, client: CompletionClient) -> None:
        self._client = client

    def plan_queries(self, theme: str, roles: Iterable[str] | None = None) -> list[PlannedQuery]:
        """Expand a theme into one query per role the deck needs covered."""
        wanted = tuple(roles) if roles is not None else _FALLBACK_ROLES
        theme = theme.strip()
        if not theme:
            return []

        return [PlannedQuery(query_text=f"{theme} {role}", purpose=role) for role in wanted]

    def select_candidates(self, query: str, cards: Iterable[Card]) -> list[Card]:
        """Drop everything the deck's format and color identity rule out."""
        constraints: Constraints = infer_constraints(query, self._client)

        kept: list[Card] = []
        for card in cards:
            legality = {fmt.title(): status for fmt, status in card.legalities.items()}
            if legality.get(constraints.deck_format) != "legal":
                continue
            if not constraints.permits(card.colors):
                continue
            kept.append(card)
        return kept
