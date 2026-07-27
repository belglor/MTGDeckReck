"""Print a curated recommendation to the terminal.

Curation's shared display, mirroring `retrieve/render.py`. Pure display: the
recommendation is a flat `list[CuratedCard]` ([ADR 0024]) and this is where
the grouping by role happens ([ADR 0005]) — the one place it happens, so
`role` and its bucket cannot drift apart.
"""

from __future__ import annotations

import polars as pl

from mtg_rag.curate.recommendation import CuratedCard


def print_recommendation(recommendation: list[CuratedCard], rows: pl.DataFrame) -> None:
    """Print `recommendation`, grouped under role headings, hydrated from `rows`.

    A card whose `oracle_id` is missing from `rows` is skipped, not crashed on
    — mirroring `print_pool`'s guard. Role order is first-appearance order in
    `recommendation`, kept in a `dict` so it is stable across runs regardless
    of string-hash randomization; a role left with no surviving cards never
    gets a heading, so a gap stays visible rather than padded.
    """
    by_id = {row["oracle_id"]: row for row in rows.iter_rows(named=True)}

    by_role: dict[str, list[CuratedCard]] = {}
    for card in recommendation:
        if card.oracle_id not in by_id:
            continue
        by_role.setdefault(card.role, []).append(card)

    for role, cards in by_role.items():
        print(f"== {role} ==")
        for position, card in enumerate(cards, start=1):
            row = by_id[card.oracle_id]
            cost = row["mana_cost"] or ""
            print(f"{position:>3}. {row['name']}  {cost}")
            print(f"     {row['type_line']}")
            print(f"     {card.rationale}")
