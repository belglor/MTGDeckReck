"""Turn the curation model's raw text into a validated recommendation, or raise.

The boundary [ADR 0024] puts between the model and the rest of the app: parse
the text as JSON, require a flat list of `{oracle_id, role, rationale}` objects,
and build a `list[CuratedCard]` — letting `CuratedCard.__post_init__` reject
empty or whitespace-only `role`/`rationale`. The one check the schema cannot do
by itself is closed-vocabulary: every `oracle_id` must be a member of the
retrieved pool, checked here against `pool_ids`. Anything that does not
validate raises.

Mirrors `plan/parse.py`: never scrape a card out of prose, never keep only the
entries that parsed.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any, cast

from mtg_rag.curate.recommendation import CuratedCard

#: The keys a recommendation entry must carry — exactly these, no more, no
#: fewer ([ADR 0024]).
_REQUIRED_KEYS = frozenset({"oracle_id", "role", "rationale"})


class MalformedRecommendationError(ValueError):
    """The model's output is not a validatable recommendation.

    A distinct type so curation's retry/raise loop can catch the malformed
    case on its own rather than swallowing every `ValueError`.
    """


def parse_recommendation(text: str, *, pool_ids: Collection[str]) -> list[CuratedCard]:
    """Parse the model's raw text into a `list[CuratedCard]`, or raise.

    Raises `MalformedRecommendationError` on anything that does not validate:
    non-JSON, prose wrapped around the JSON, a non-list top level, an entry
    that is not an object, a missing or extra key, a non-string value, an
    empty/whitespace-only `role` or `rationale`, or an `oracle_id` not in
    `pool_ids`. An empty list is a valid recommendation — curation may select
    no candidate from the pool. A pool id the model omits does not raise
    either; not every candidate makes the deck ([ADR 0024]).
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as err:
        raise MalformedRecommendationError(f"curation output is not valid JSON: {err}") from err

    if not isinstance(payload, list):
        raise MalformedRecommendationError(
            f"curation output must be a JSON list, got {type(payload).__name__}"
        )
    entries = cast("list[Any]", payload)

    return [_entry_to_card(index, entry, pool_ids) for index, entry in enumerate(entries)]


def _entry_to_card(index: int, entry: Any, pool_ids: Collection[str]) -> CuratedCard:
    if not isinstance(entry, dict):
        raise MalformedRecommendationError(
            f"recommendation entry {index} must be an object, got {type(entry).__name__}"
        )

    obj = cast("dict[str, Any]", entry)
    keys = frozenset(obj)
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise MalformedRecommendationError(
            f"recommendation entry {index} is missing key(s): {', '.join(sorted(missing))}"
        )
    extra = keys - _REQUIRED_KEYS
    if extra:
        raise MalformedRecommendationError(
            f"recommendation entry {index} has unexpected key(s): {', '.join(sorted(extra))}"
        )

    for key in _REQUIRED_KEYS:
        if not isinstance(obj[key], str):
            got = type(obj[key]).__name__
            raise MalformedRecommendationError(
                f"recommendation entry {index} {key} must be a string, got {got}"
            )

    oracle_id = obj["oracle_id"]
    if oracle_id not in pool_ids:
        raise MalformedRecommendationError(
            f"recommendation entry {index} oracle_id {oracle_id!r} is not in the retrieved pool"
        )

    try:
        return CuratedCard(oracle_id=oracle_id, role=obj["role"], rationale=obj["rationale"])
    except ValueError as err:
        raise MalformedRecommendationError(f"recommendation entry {index}: {err}") from err
