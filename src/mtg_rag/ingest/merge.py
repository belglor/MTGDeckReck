"""Collapse a card's many printings into the one record the corpus stores.

Ingestion reads `default_cards`, which ships one object per printing ([ADR 0016]),
so a card arrives once per time it was printed — 116,138 objects for 38,320
cards. [ADR 0002] says one card is one record, and this module is where that
holds: many printings in, exactly one record per `oracle_id` out.

Most fields make the choice moot. Measured across every printing, `oracle_text`,
`type_line` and legality vary on **zero** cards — Scryfall applies current oracle
text to every printing, and legality is a property of the card rather than the
cardboard. What genuinely differs is `rarity` (3,316 cards), `flavor_text` (3,550
carry more than one), and the set, release date, `games` and prices, which differ
on nearly every reprint.

Two rules cover that. A **representative printing** — the most recent — supplies
every single-valued field, so a row describes one physical card rather than a
composite of several. **Flavor text is the exception**: it comes from the most
recent printing that actually has any. Taking it from the representative printing
instead would strip flavor text from 1,804 cards, because a card's newest
printing is so often a Commander-deck or promo reprint that carries none, and the
flavor channel is the sparsest of the three to begin with.

The fold is incremental rather than a group-by: only the per-card aggregate is
held, never the 116,138 records that produced it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from mtg_rag.corpus import is_real
from mtg_rag.ingest.normalize import CardRecord

#: How printings of one card are ordered, best last.
#:
#: Being a real printing outranks being a recent one: otherwise a memorabilia
#: reprint represents the card, and the structural predicate then discards the
#: card itself ([ADR 0016]). A card whose printings are all non-cards still
#: merges, and is excluded downstream.
#:
#: Then release date, with set code breaking ties so repeated runs agree. A
#: printing with no date sorts last — it cannot be shown to be the most recent.
type _Order = tuple[bool, str, str]


def _order(record: CardRecord) -> _Order:
    return (
        is_real(record.layout, record.set_type),
        record.released_at or "",
        record.set_code or "",
    )


def _flavor_of(record: CardRecord) -> str | None:
    """A printing's flavor text, or None when it has none worth taking.

    `normalize_card` already maps absent and empty values to None; the strip
    guards the whitespace-only case, which would otherwise let a blank donor
    outrank a real one.
    """
    text = record.flavor_text
    if text is None or not text.strip():
        return None
    return text


def merge_printings(records: Iterable[CardRecord]) -> list[CardRecord]:
    """One record per `oracle_id`, ordered by it.

    Ordering the result makes the corpus a function of its input alone, rather
    than of the order Scryfall happened to ship printings in — two runs over one
    snapshot produce the same parquet.
    """
    representatives: dict[str, tuple[_Order, CardRecord]] = {}
    flavors: dict[str, tuple[_Order, str]] = {}

    for record in records:
        order = _order(record)

        chosen = representatives.get(record.oracle_id)
        if chosen is None or order > chosen[0]:
            representatives[record.oracle_id] = (order, record)

        flavor = _flavor_of(record)
        if flavor is not None:
            donor = flavors.get(record.oracle_id)
            if donor is None or order > donor[0]:
                flavors[record.oracle_id] = (order, flavor)

    merged: list[CardRecord] = []
    for oracle_id, (_, record) in sorted(representatives.items()):
        donor = flavors.get(oracle_id)
        merged.append(replace(record, flavor_text=donor[1] if donor is not None else None))
    return merged
