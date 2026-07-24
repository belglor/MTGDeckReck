"""The golden-set schema: parse `golden.toml` into typed cases, and refuse the
malformed ones before a run starts.

A case is `(query, predicate, one or more constraint sets)` ([ADR 0020]). Two
constraint sets make it a constraint-interaction case; that is not a separate
kind and must not become one.

Validation is split in two because the two halves need different things.
`load_cases` is structural and needs only the file, so it runs in the test suite
with no corpus. `validate_against_corpus` needs the frame, and exists so a
mistyped keyword fails in milliseconds rather than after a model load — a
predicate nothing satisfies has a zero base rate and therefore no defined lift,
which is a malformed case rather than a result.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

from mtg_rag.evals.config import GOLDEN_NAME, PredicateKind
from mtg_rag.evals.metrics import base_rate
from mtg_rag.evals.predicates import predicate_expr
from mtg_rag.retrieve.filters import Constraints

#: What a `[[case]]` table may contain. An unlisted key is a typo — most likely
#: a misspelled `rationale` — and silently ignoring it would drop the one field
#: a reader cannot reconstruct.
CASE_FIELDS = frozenset({"id", "query", "rationale", "predicate", "constraints"})

#: What one entry of `constraints` may contain, mirroring `Constraints`.
CONSTRAINT_FIELDS = frozenset({"format", "colors", "platform"})


class MalformedCaseError(ValueError):
    """A case is missing something, or names something that cannot be run."""


@dataclass(frozen=True, slots=True)
class Predicate:
    """The property a case expects the pool to be enriched for."""

    kind: PredicateKind
    value: str

    def expr(self) -> pl.Expr:
        return predicate_expr(self.kind, self.value)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One golden-set case.

    `rationale` is required and says why this predicate is defensible ground
    truth for this query. It is the only field a later reader cannot reconstruct
    from the others, which is exactly why it is not optional.
    """

    id: str
    query: str
    rationale: str
    predicate: Predicate
    constraints: tuple[Constraints, ...]


def golden_path() -> Path:
    """The shipped golden set, next to this module."""
    return Path(__file__).parent / GOLDEN_NAME


def _color_identity(colors: str | None) -> frozenset[str] | None:
    """`colors` as a constraint.

    Absent is unconstrained; `""` is colorless-only. `Constraints` puts that
    distinction in the type rather than a sentinel, so parsing must preserve it.
    """
    if colors is None:
        return None
    return frozenset(colors.strip().upper())


def _constraints(case_id: str, entries: Sequence[Any]) -> tuple[Constraints, ...]:
    if not entries:
        raise MalformedCaseError(f"case {case_id!r} has no constraints; it needs at least one")
    parsed: list[Constraints] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise MalformedCaseError(f"case {case_id!r} has a constraint that is not a table")
        table = cast("dict[str, Any]", entry)
        unknown = set(table) - CONSTRAINT_FIELDS
        if unknown:
            raise MalformedCaseError(
                f"case {case_id!r} constraint has unknown field(s): {', '.join(sorted(unknown))}"
            )
        format_name = table.get("format")
        if not isinstance(format_name, str) or not format_name:
            raise MalformedCaseError(f"case {case_id!r} constraint has no format")
        colors = table.get("colors")
        if colors is not None and not isinstance(colors, str):
            raise MalformedCaseError(f"case {case_id!r} constraint has non-string colors")
        platform = table.get("platform")
        if platform is not None and not isinstance(platform, str):
            raise MalformedCaseError(f"case {case_id!r} constraint has non-string platform")
        fields: dict[str, Any] = {
            "format_name": format_name,
            "color_identity": _color_identity(colors),
        }
        if platform is not None:
            fields["platform"] = platform
        parsed.append(Constraints(**fields))
    return tuple(parsed)


def _predicate(case_id: str, table: Any) -> Predicate:
    """Read the predicate table, whose single key *is* the kind.

    Naming the kind separately from the value would let the two drift; here a
    case cannot claim `keyword` while carrying an oracle-text regex.
    """
    if not isinstance(table, dict):
        raise MalformedCaseError(f"case {case_id!r} has no predicate table")
    entries = cast("dict[str, Any]", table)
    if len(entries) != 1:
        raise MalformedCaseError(
            f"case {case_id!r} predicate must name exactly one kind, got {len(entries)}"
        )
    kind, value = next(iter(entries.items()))
    if not isinstance(value, str) or not value:
        raise MalformedCaseError(f"case {case_id!r} predicate {kind!r} has no value")
    # Builds the expression now so an unknown kind raises here, at load, rather
    # than once a run is already underway.
    predicate_expr(kind, value)
    return Predicate(kind=cast("PredicateKind", kind), value=value)


def _case(raw: Mapping[str, Any]) -> EvalCase:
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise MalformedCaseError("a case has no id")
    unknown = set(raw) - CASE_FIELDS
    if unknown:
        raise MalformedCaseError(
            f"case {case_id!r} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    for field in ("query", "rationale"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise MalformedCaseError(f"case {case_id!r} has no {field}")
    return EvalCase(
        id=case_id,
        query=cast("str", raw["query"]),
        rationale=cast("str", raw["rationale"]),
        predicate=_predicate(case_id, raw.get("predicate")),
        constraints=_constraints(case_id, cast("Sequence[Any]", raw.get("constraints") or [])),
    )


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    """Parse and structurally validate a golden set. Raises on anything malformed."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("case")
    if not isinstance(raw_cases, list):
        raise MalformedCaseError(f"{path} has no [[case]] tables")
    cases = tuple(
        _case(cast("Mapping[str, Any]", entry))
        for entry in cast("list[Any]", raw_cases)
        if isinstance(entry, dict)
    )
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise MalformedCaseError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
    return cases


def validate_against_corpus(cases: Iterable[EvalCase], frame: pl.DataFrame) -> None:
    """Refuse any case whose predicate no allowed card satisfies.

    A zero base rate makes lift undefined, so such a case can only ever report
    nothing. Far more often it means a typo — `"Madnes"`, or a lowercase
    `"madness"` against a column that carries `"Madness"` — and failing here
    costs milliseconds where failing mid-run costs a model load.
    """
    for case in cases:
        predicate = case.predicate.expr()
        for constraints in case.constraints:
            if base_rate(frame, constraints, predicate) > 0.0:
                continue
            raise MalformedCaseError(
                f"case {case.id!r}: no card allowed by {constraints.format_name}"
                f"/{constraints.color_identity or 'any'}/{constraints.platform} satisfies "
                f"{case.predicate.kind}={case.predicate.value!r} — check the spelling "
                f"(keywords are capitalised, e.g. 'Madness')"
            )
