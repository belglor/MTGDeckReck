"""Print a fused candidate pool to the terminal.

The retrieval CLIs' shared display. Kept out of `pool.py`, which stays pure — it
builds and hydrates a pool as data and touches no stdout — and out of either
`__main__`, so both `just retrieve` and the `just plan` that now drives it print
an identical pool without a second copy of the format.
"""

from __future__ import annotations

import polars as pl

from mtg_rag.retrieve.fusion import Candidate


def print_pool(pool: list[Candidate], rows: pl.DataFrame, *, explain: bool) -> None:
    by_id = {row["oracle_id"]: row for row in rows.iter_rows(named=True)}
    for position, candidate in enumerate(pool, start=1):
        card = by_id.get(candidate.oracle_id)
        if card is None:  # pragma: no cover - hydration drops unknown ids
            continue
        cost = card["mana_cost"] or ""
        print(f"{position:>3}. {card['name']}  {cost}")
        print(f"     {card['type_line']}   (score {candidate.score:.4f})")
        if explain:
            for source in candidate.sources:
                # Distance is shown, never ranked on: it says whether this
                # channel was confident, which rank alone cannot ([ADR 0008]).
                print(
                    f"       - {source.channel:<7} rank {source.rank:<3} "
                    f"dist {source.distance:.3f}  {source.purpose}"
                )
