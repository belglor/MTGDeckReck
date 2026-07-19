"""Download Scryfall's `oracle_cards` bulk snapshot and normalize it to a corpus.

The projection here is deliberately narrow: the fields the spec actually uses,
flattened, one row per card. Storage format and identity key are settled in
[ADR 0009] and [ADR 0010]; the constraint that shapes this module most is
[ADR 0002] — one card is one record, so a multi-face card's faces are joined
rather than split across rows.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import httpx
import polars as pl

BULK_INDEX_URL = "https://api.scryfall.com/bulk-data"
DEFAULT_BULK_TYPE = "oracle_cards"

#: Scryfall asks API clients to identify themselves; requests without a
#: User-Agent are rejected outright.
USER_AGENT = "MTGDeckReck/0.1 (https://github.com/belglor/MTGDeckReck)"

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
class BulkEntry:
    """One entry from Scryfall's bulk-data index."""

    bulk_type: str
    updated_at: str
    download_uri: str
    size: int


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Provenance sidecar recording which bulk build produced the corpus.

    This is what makes a re-run idempotent: if the live index still reports the
    `updated_at` we last ingested, there is nothing to do.
    """

    bulk_type: str
    updated_at: str
    download_uri: str
    row_count: int
    ingested_at: str

    def write(self, path: Path) -> None:
        payload = {
            "bulk_type": self.bulk_type,
            "updated_at": self.updated_at,
            "download_uri": self.download_uri,
            "row_count": self.row_count,
            "ingested_at": self.ingested_at,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self | None:
        """Load a sidecar, or return None if it is absent or unusable.

        A corrupt sidecar means we cannot trust our own provenance, so the
        honest response is to rebuild rather than to crash or to assume.
        """
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        data = cast("dict[str, Any]", payload)
        try:
            return cls(
                bulk_type=str(data["bulk_type"]),
                updated_at=str(data["updated_at"]),
                download_uri=str(data["download_uri"]),
                row_count=int(data["row_count"]),
                ingested_at=str(data["ingested_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


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


def _str_or_none(raw: Mapping[str, Any], key: str) -> str | None:
    value: Any = raw.get(key)
    return value if isinstance(value, str) and value else None


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
    return [item for item in cast("list[Any]", value) if isinstance(item, str)]


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
    return {k: v for k, v in items.items() if isinstance(k, str) and isinstance(v, str)}


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
        layout=_str_or_none(raw, "layout") or "unknown",
        rarity=_str_or_none(raw, "rarity"),
        set_code=_str_or_none(raw, "set"),
        set_name=_str_or_none(raw, "set_name"),
        set_type=_str_or_none(raw, "set_type"),
        released_at=_str_or_none(raw, "released_at"),
        games=_str_list(raw, "games"),
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


# --- I/O -------------------------------------------------------------------


def fetch_bulk_entry(client: httpx.Client, bulk_type: str = DEFAULT_BULK_TYPE) -> BulkEntry:
    """Look up one entry in Scryfall's bulk-data index."""
    response = client.get(BULK_INDEX_URL)
    response.raise_for_status()
    payload: Any = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected bulk-data index shape from {BULK_INDEX_URL}")
    entries: Any = cast("dict[str, Any]", payload).get("data")
    if not isinstance(entries, list):
        raise ValueError(f"unexpected bulk-data index shape from {BULK_INDEX_URL}")

    for entry in cast("list[Any]", entries):
        if not isinstance(entry, dict):
            continue
        record = cast("dict[str, Any]", entry)
        if record.get("type") != bulk_type:
            continue
        # Prefer the gzipped NDJSON variant: it streams line by line, where the
        # plain .json variant is a single ~180 MB array.
        uri = _str_or_none(record, "jsonl_download_uri") or _str_or_none(record, "download_uri")
        updated_at = _str_or_none(record, "updated_at")
        if uri is None or updated_at is None:
            raise ValueError(f"bulk entry {bulk_type!r} is missing a download URI or timestamp")
        return BulkEntry(
            bulk_type=bulk_type,
            updated_at=updated_at,
            download_uri=uri,
            size=_int_or_none(record.get("size")) or 0,
        )

    raise ValueError(f"no bulk-data entry of type {bulk_type!r}")


def download(client: httpx.Client, uri: str, dest: Path) -> None:
    """Stream a bulk file to disk without holding it in memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", uri) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)


def stream_cards(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield card objects from a gzipped NDJSON bulk file, one at a time.

    Also tolerates the plain-JSON array variant, since `fetch_bulk_entry` falls
    back to it if Scryfall ever stops publishing the NDJSON URI.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        first = handle.readline()
        if not first.strip():
            return
        if first.lstrip().startswith("["):
            handle.seek(0)
            payload: Any = json.load(handle)
            if isinstance(payload, list):
                for card in cast("list[Any]", payload):
                    if isinstance(card, dict):
                        yield cast("Mapping[str, Any]", card)
            return
        for line in (first, *handle):
            stripped = line.strip().rstrip(",")
            if not stripped or stripped in {"[", "]"}:
                continue
            parsed: Any = json.loads(stripped)
            if isinstance(parsed, dict):
                yield cast("Mapping[str, Any]", parsed)


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=httpx.Timeout(30.0, read=300.0),
        follow_redirects=True,
    )


def should_skip(
    snapshot: Snapshot | None,
    entry: BulkEntry,
    corpus_path: Path,
    *,
    force: bool,
) -> bool:
    """Decide whether the corpus is already current.

    Requires the corpus file to actually exist, not just the sidecar to agree —
    deleting `data/` must fully restore on the next run.
    """
    if force or snapshot is None:
        return False
    if not corpus_path.exists():
        return False
    return snapshot.bulk_type == entry.bulk_type and snapshot.updated_at == entry.updated_at


def summarize(frame: pl.DataFrame) -> Sequence[str]:
    """Lines describing what was ingested, for the CLI to print."""
    multi_face = frame.filter(pl.col("oracle_text").str.contains(FACE_SEPARATOR, literal=True))
    with_flavor = frame.filter(pl.col("flavor_text").is_not_null())
    return [
        f"cards:        {frame.height:,}",
        f"columns:      {len(frame.columns)}",
        f"with flavor:  {with_flavor.height:,} ({with_flavor.height / frame.height:.0%})",
        f"multi-face:   {multi_face.height:,}",
    ]
