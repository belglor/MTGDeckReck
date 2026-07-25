"""Collapse a card's many printings into the one record the corpus stores.

Ingestion reads `default_cards`, one object per printing ([ADR 0016]); this is
where [ADR 0002]'s one-card-one-record holds — many printings in, one record per
`oracle_id` out. Oracle text, type line and legality don't vary across a card's
printings, so the choice only bites on the fields that do.

A **representative printing** (the most recent) supplies every single-valued
field, so a row is one physical card, not a composite of several. Two fields are
taken differently: **flavor text** from the most recent printing that *has* any
(the newest printing is often a reprint carrying none, and flavor is already the
sparsest channel); **platforms** as the union across printings, the only level at
which "can I play this on Arena?" has an answer ([ADR 0018]).

The fold is incremental — only the per-card aggregate is held, never every
printing that produced it.
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
    platforms: dict[str, set[str]] = {}

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

        platforms.setdefault(record.oracle_id, set()).update(record.platforms)

    merged: list[CardRecord] = []
    for oracle_id, (_, record) in sorted(representatives.items()):
        donor = flavors.get(oracle_id)
        merged.append(
            replace(
                record,
                flavor_text=donor[1] if donor is not None else None,
                platforms=sorted(platforms[oracle_id]),
            )
        )
    return merged
