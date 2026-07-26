"""Turn the planner model's raw text into a validated plan, or raise.

The boundary [ADR 0022] puts between the model and the rest of the app: parse
the text as JSON, require a list of `{query_text, purpose}` objects, and build a
`list[PlannedQuery]` — letting `PlannedQuery.__post_init__` ([ADR 0004]) reject
empty or whitespace-only fields. Anything that does not validate raises.

`transformers` enforces no schema, so the model may return bad JSON, prose
around the JSON, extra keys, or empty fields. None of it is salvaged: never
scrape a query out of prose (a phrasing drift would become a silent misparse,
the failure the schema exists to kill — [ADR 0004]) and never keep only the
entries that parsed. A loud failure is recoverable; a quietly-wrong plan flows
downstream into a pool no curation can rescue.
"""

from __future__ import annotations

import json
from typing import Any, cast

from mtg_rag.plan.query import PlannedQuery

#: The keys a plan entry must carry — exactly these, no more, no fewer. `purpose`
#: is required, not decoration: it travels to curation as the card's starting
#: role hypothesis ([ADR 0005]).
_REQUIRED_KEYS = frozenset({"query_text", "purpose"})


class MalformedPlanError(ValueError):
    """The model's output is not a validatable plan.

    A distinct type so the planner's retry/raise loop can catch the malformed
    case on its own rather than swallowing every `ValueError`.
    """


def parse_plan(text: str) -> list[PlannedQuery]:
    """Parse the model's raw text into a `list[PlannedQuery]`, or raise.

    Raises `MalformedPlanError` on anything that does not validate: non-JSON,
    prose wrapped around the JSON, a non-list top level, an empty list, an entry
    that is not an object, a missing or extra key, a non-string value, or an
    empty/whitespace-only field.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as err:
        raise MalformedPlanError(f"planner output is not valid JSON: {err}") from err

    if not isinstance(payload, list):
        raise MalformedPlanError(
            f"planner output must be a JSON list, got {type(payload).__name__}"
        )
    entries = cast("list[Any]", payload)
    if not entries:
        raise MalformedPlanError("planner output is an empty list; a plan needs at least one query")

    return [_entry_to_query(index, entry) for index, entry in enumerate(entries)]


def _entry_to_query(index: int, entry: Any) -> PlannedQuery:
    if not isinstance(entry, dict):
        raise MalformedPlanError(
            f"plan entry {index} must be an object, got {type(entry).__name__}"
        )

    obj = cast("dict[str, Any]", entry)
    keys = frozenset(obj)
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise MalformedPlanError(
            f"plan entry {index} is missing key(s): {', '.join(sorted(missing))}"
        )
    extra = keys - _REQUIRED_KEYS
    if extra:
        raise MalformedPlanError(
            f"plan entry {index} has unexpected key(s): {', '.join(sorted(extra))}"
        )

    for key in _REQUIRED_KEYS:
        if not isinstance(obj[key], str):
            raise MalformedPlanError(
                f"plan entry {index} {key} must be a string, got {type(obj[key]).__name__}"
            )

    try:
        return PlannedQuery(query_text=obj["query_text"], purpose=obj["purpose"])
    except ValueError as err:
        raise MalformedPlanError(f"plan entry {index}: {err}") from err
