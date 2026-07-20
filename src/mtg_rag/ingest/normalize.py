"""Project a raw Scryfall card object into a `CardRecord`.

The projection here is deliberately narrow: the fields the spec actually uses,
flattened, one row per card. Storage format and identity key are settled in
[ADR 0009] and [ADR 0010]; the constraint that shapes this module most is
[ADR 0002] — one card is one record, so a multi-face card's faces are joined
rather than split across rows.

Two preprocessing passes run underneath every field read: Unicode NFC
normalization on text, and lowercasing on the fields used for exact-match
filtering. Scryfall's export happens to already be consistent on both axes,
but nothing about the API guarantees it, and the failure mode of trusting it
is silent — a decomposed accent makes two copies of the same name embed
differently, and a `Legal` slipping in among `legal` values would drop a
card out of [ADR 0001]'s filters without an error anywhere.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import polars as pl

#: Separator for text joined across a card's faces. Newlines keep the halves
#: visually distinct in oracle and flavor text, matching how Scryfall renders
#: split cards.
FACE_SEPARATOR = "\n//\n"

#: Mana costs are joined inline instead, since a newline inside a cost string
#: would be nonsense.
COST_SEPARATOR = " // "


class MalformedCardError(ValueError):
    """A card object is missing something we cannot proceed without."""


@dataclass(frozen=True, slots=True)
class CardRecord:
    """One card, projected to the fields this project uses."""

    oracle_id: str
    name: str
    oracle_text: str | None
    flavor_text: str | None
    type_line: str | None
    mana_cost: str | None
    cmc: float | None
    colors: list[str]
    color_identity: list[str]
    keywords: list[str]
    power: str | None
    toughness: str | None
    loyalty: str | None
    layout: str
    rarity: str | None
    set_code: str | None
    set_name: str | None
    set_type: str | None
    released_at: str | None
    games: list[str]
    reserved: bool
    edhrec_rank: int | None
    price_usd: float | None
    price_eur: float | None
    price_tix: float | None
    legalities: Mapping[str, str]


# --- typed accessors -------------------------------------------------------
# json.loads hands back Any, so every read is narrowed explicitly rather than
# cast. Scryfall omits absent fields instead of nulling them, and omits
# different fields for different layouts, so nothing may be assumed present.


def _normalize_text(value: str) -> str:
    """NFC-normalize so visually identical strings compare and embed alike."""
    return unicodedata.normalize("NFC", value)


def _str_or_none(raw: Mapping[str, Any], key: str) -> str | None:
    value: Any = raw.get(key)
    if not isinstance(value, str) or not value:
        return None
    return _normalize_text(value)


def _lower_or_none(raw: Mapping[str, Any], key: str) -> str | None:
    """Like `_str_or_none`, for fields matched exactly rather than displayed."""
    value = _str_or_none(raw, key)
    return value.lower() if value is not None else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _str_list(raw: Mapping[str, Any], key: str) -> list[str]:
    value: Any = raw.get(key)
    if not isinstance(value, list):
        return []
    return [_normalize_text(item) for item in cast("list[Any]", value) if isinstance(item, str)]


def _lower_str_list(raw: Mapping[str, Any], key: str) -> list[str]:
    """Like `_str_list`, for fields matched exactly rather than displayed."""
    return [item.lower() for item in _str_list(raw, key)]


def _faces(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value: Any = raw.get("card_faces")
    if not isinstance(value, list):
        return []
    return [
        cast("Mapping[str, Any]", face)
        for face in cast("list[Any]", value)
        if isinstance(face, dict)
    ]


def _joined(raw: Mapping[str, Any], key: str, separator: str = FACE_SEPARATOR) -> str | None:
    """Read a text field, falling back to joining it across the card's faces.

    Split and modal double-faced cards carry no top-level `oracle_text` or
    `flavor_text` at all — that text exists only under `card_faces`. Joining
    keeps the card in one record, per [ADR 0002]. Empty face values are dropped
    so a costless land face does not leave a dangling separator.
    """
    top = _str_or_none(raw, key)
    if top is not None:
        return top
    parts = [text for face in _faces(raw) if (text := _str_or_none(face, key)) is not None]
    return separator.join(parts) if parts else None


def _from_faces(raw: Mapping[str, Any], key: str) -> str | None:
    """Read a field that belongs to a single face, e.g. power or loyalty."""
    top = _str_or_none(raw, key)
    if top is not None:
        return top
    for face in _faces(raw):
        value = _str_or_none(face, key)
        if value is not None:
            return value
    return None


def _legalities(raw: Mapping[str, Any]) -> Mapping[str, str]:
    value: Any = raw.get("legalities")
    if not isinstance(value, dict):
        return {}
    items = cast("dict[Any, Any]", value)
    return {
        k.lower(): v.lower()
        for k, v in items.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _prices(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value: Any = raw.get("prices")
    if not isinstance(value, dict):
        return {}
    priced = cast("dict[Any, Any]", value)
    return {k: v for k, v in priced.items() if isinstance(k, str)}


# --- the projection --------------------------------------------------------


def normalize_card(raw: Mapping[str, Any]) -> CardRecord:
    """Project one Scryfall card object into a `CardRecord`.

    Raises `MalformedCardError` when `oracle_id` is missing — that is the join
    key for every vector ([ADR 0010]), so a record without one is not something
    to silently drop or paper over.
    """
    oracle_id = _str_or_none(raw, "oracle_id")
    if oracle_id is None:
        name = _str_or_none(raw, "name") or "<unnamed>"
        raise MalformedCardError(f"card {name!r} has no oracle_id")

    prices = _prices(raw)

    return CardRecord(
        oracle_id=oracle_id,
        name=_str_or_none(raw, "name") or "",
        oracle_text=_joined(raw, "oracle_text"),
        flavor_text=_joined(raw, "flavor_text"),
        type_line=_joined(raw, "type_line", COST_SEPARATOR),
        mana_cost=_joined(raw, "mana_cost", COST_SEPARATOR),
        cmc=_float_or_none(raw.get("cmc")),
        colors=_str_list(raw, "colors"),
        color_identity=_str_list(raw, "color_identity"),
        keywords=_str_list(raw, "keywords"),
        power=_from_faces(raw, "power"),
        toughness=_from_faces(raw, "toughness"),
        loyalty=_from_faces(raw, "loyalty"),
        layout=_lower_or_none(raw, "layout") or "unknown",
        rarity=_lower_or_none(raw, "rarity"),
        set_code=_lower_or_none(raw, "set"),
        set_name=_str_or_none(raw, "set_name"),
        set_type=_lower_or_none(raw, "set_type"),
        released_at=_str_or_none(raw, "released_at"),
        games=_lower_str_list(raw, "games"),
        reserved=raw.get("reserved") is True,
        edhrec_rank=_int_or_none(raw.get("edhrec_rank")),
        price_usd=_float_or_none(prices.get("usd")),
        price_eur=_float_or_none(prices.get("eur")),
        price_tix=_float_or_none(prices.get("tix")),
        legalities=_legalities(raw),
    )


def build_frame(records: Iterable[CardRecord]) -> pl.DataFrame:
    """Assemble records into the corpus frame, flattening legalities.

    Each format becomes its own `legal_<format>` string column. The column set
    is the union across all records rather than a hardcoded list, so a format
    Scryfall adds later appears without a code change. Values stay strings
    because legality has four states, not two — collapsing to a boolean would
    lose the distinction between `banned` and `restricted`.
    """
    materialized = list(records)
    if not materialized:
        raise ValueError("no cards to write — refusing to build an empty corpus")

    formats = sorted({fmt for record in materialized for fmt in record.legalities})

    rows: list[dict[str, Any]] = []
    for record in materialized:
        row: dict[str, Any] = {
            "oracle_id": record.oracle_id,
            "name": record.name,
            "oracle_text": record.oracle_text,
            "flavor_text": record.flavor_text,
            "type_line": record.type_line,
            "mana_cost": record.mana_cost,
            "cmc": record.cmc,
            "colors": record.colors,
            "color_identity": record.color_identity,
            "keywords": record.keywords,
            "power": record.power,
            "toughness": record.toughness,
            "loyalty": record.loyalty,
            "layout": record.layout,
            "rarity": record.rarity,
            "set_code": record.set_code,
            "set_name": record.set_name,
            "set_type": record.set_type,
            "released_at": record.released_at,
            "games": record.games,
            "reserved": record.reserved,
            "edhrec_rank": record.edhrec_rank,
            "price_usd": record.price_usd,
            "price_eur": record.price_eur,
            "price_tix": record.price_tix,
        }
        for fmt in formats:
            row[f"legal_{fmt}"] = record.legalities.get(fmt)
        rows.append(row)

    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("released_at").str.to_date(strict=False)
    )

    duplicates = frame.height - frame["oracle_id"].n_unique()
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate oracle_id values — the corpus key must be unique (ADR 0010)"
        )
    return frame
