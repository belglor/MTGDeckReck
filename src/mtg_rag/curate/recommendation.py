"""What curation emits: one card, its role, and why it fits.

Data only — no prompt, no model, no template. [ADR 0024] settles the shape: a
flat `[{oracle_id, role, rationale}]`, one `CuratedCard` per entry, that the
renderer groups by `role` for display ([ADR 0005]).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CuratedCard:
    """One card curation selected, its role, and the theme-fit argument for it.

    `role` and `rationale` are required and non-empty; whitespace counts as
    empty, mirroring `PlannedQuery`. `oracle_id` is not checked here — it is
    closed-vocabulary against the retrieved pool, and that check needs the
    pool, so it belongs to the parser at the boundary ([ADR 0024]).
    """

    oracle_id: str
    role: str
    rationale: str

    def __post_init__(self) -> None:
        for field, value in (("role", self.role), ("rationale", self.rationale)):
            if not value.strip():
                raise ValueError(f"{field} must be a non-empty string, got {value!r}")
