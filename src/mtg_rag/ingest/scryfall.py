"""Download Scryfall's `default_cards` bulk snapshot to a local corpus.

Network fetch, streaming, and idempotency live here. The projection from a raw
card object to a `CardRecord` — the part that decides what a card *means* to
this project — lives in `mtg_rag.ingest.normalize`; see [ADR 0002] for why
that module joins multi-face cards rather than splitting them across rows.

The snapshot holds one object per *printing* ([ADR 0016]), so what streams out
of here is printings, not cards. `mtg_rag.ingest.merge` collapses them.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import httpx
import polars as pl

from mtg_rag.ingest.config import (
    BULK_INDEX_URL,
    BULK_TYPE,
    CORPUS_LANGUAGE,
    FACE_SEPARATOR,
    USER_AGENT,
)


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


# --- I/O -------------------------------------------------------------------
# json.loads hands back Any, so every read from the bulk-data index is
# narrowed explicitly rather than cast.


def _str_or_none(raw: Mapping[str, Any], key: str) -> str | None:
    value: Any = raw.get(key)
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def is_english(raw: Mapping[str, Any]) -> bool:
    """Whether a printing is the English one we want in the corpus.

    A missing `lang` counts as English. Scryfall always sets it, so this only
    decides what happens to a malformed object — and keeping a card we should
    have dropped is a far smaller error than dropping a real card on the basis
    of an absent field.
    """
    lang: Any = raw.get("lang")
    if not isinstance(lang, str):
        return True
    return lang == CORPUS_LANGUAGE


def fetch_bulk_entry(client: httpx.Client, bulk_type: str = BULK_TYPE) -> BulkEntry:
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
        # plain .json variant is a single ~560 MB array. It is also a quarter of
        # the bytes — 72.6 MB against 557.6 MB.
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
