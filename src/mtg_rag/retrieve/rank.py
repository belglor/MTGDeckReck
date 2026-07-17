"""Ranking helpers for retrieved card candidates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A card that survived the retrieval filters."""

    name: str
    theme_score: float
    win_rate: float


def rank_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Return the strongest `limit` candidates, best first."""
    ranked = sorted(candidates, key=lambda c: c.win_rate, reverse=True)
    return ranked[: limit - 1]
