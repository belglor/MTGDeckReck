"""Pair a retrieved candidate with its corpus row, ready for curation's prompt.

The hand-off `curate` expects, kept out of `prompt.py` so that stays a pure
string transform with no corpus knowledge ([ADR 0025]'s scope note). Shared by
`just plan` and the defect sweep ([ADR 0026]), which both need the identical
assembly — the CLI owned it first, and a harness importing it from there would
drag an argparse module into a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from mtg_rag.curate.prompt import CurationCard
from mtg_rag.retrieve.fusion import Candidate


def curation_cards(pool: Sequence[Candidate], rows: pl.DataFrame) -> list[CurationCard]:
    """Pair each hydrated row with the purposes of the searches that found it.

    Driven by `rows` rather than by `pool`, because hydration drops ids the
    corpus no longer holds and this must drop them too.

    Purposes are de-duplicated but keep source order — a card found three times
    for "self-mill" says that once. They are curation's starting hypothesis for
    the card's role, not the answer ([ADR 0005]). Null corpus fields become
    empty strings, which is how the prompt knows to leave the line out.
    """
    purposes = {
        candidate.oracle_id: tuple(dict.fromkeys(source.purpose for source in candidate.sources))
        for candidate in pool
    }
    return [
        CurationCard(
            oracle_id=row["oracle_id"],
            name=row["name"],
            mana_cost=row["mana_cost"] or "",
            type_line=row["type_line"] or "",
            oracle_text=row["oracle_text"] or "",
            flavor_text=row["flavor_text"] or "",
            purposes=purposes[row["oracle_id"]],
        )
        for row in rows.iter_rows(named=True)
    ]
