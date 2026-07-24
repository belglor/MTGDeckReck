"""Tests for the golden-set schema: parsing, structural validation, and the
corpus check that refuses a predicate nothing satisfies.

The shipped `golden.toml` is parsed here too. That is cheap and catches a
hand-edited typo before a run that needs the model does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mtg_rag.evals.cases import (
    MalformedCaseError,
    golden_path,
    load_cases,
    validate_against_corpus,
)
from mtg_rag.evals.predicates import UnknownPredicateKindError
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.retrieve.config import DEFAULT_PLATFORM


def _card(name: str, *, keywords: list[str] | None = None) -> dict[str, Any]:
    return {
        "oracle_id": f"id-{name}",
        "name": name,
        "oracle_text": "Draw a card, then discard a card.",
        "type_line": "Creature — Test",
        "keywords": keywords or [],
        "color_identity": ["B"],
        "layout": "normal",
        "set_type": "expansion",
        "released_at": "2020-01-01",
        "games": ["paper"],
        "legalities": {"commander": "legal"},
    }


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    return build_frame([normalize_card(_card("connives", keywords=["Connive"]))])


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cases.toml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL = """
[[case]]
id = "connive"
query = "connive"
rationale = "Scryfall's keywords column is the ground truth."
predicate = { keyword = "Connive" }
constraints = [ { format = "commander" } ]
"""


# --- parsing ---------------------------------------------------------------


def test_a_minimal_case_loads(tmp_path: Path) -> None:
    (case,) = load_cases(_write(tmp_path, MINIMAL))
    assert case.id == "connive"
    assert case.query == "connive"
    assert case.predicate.kind == "keyword"
    assert case.predicate.value == "Connive"
    assert len(case.constraints) == 1
    assert case.constraints[0].format_name == "commander"
    assert case.constraints[0].color_identity is None
    assert case.constraints[0].platform == DEFAULT_PLATFORM


def test_case_with_two_constraint_sets_loads_as_two_runs(tmp_path: Path) -> None:
    body = """
[[case]]
id = "graveyard"
query = "graveyard recursion"
rationale = "An oracle-text mention is a proxy, fixed so it can detect movement."
predicate = { oracle_text = "(?i)graveyard" }
constraints = [ { format = "commander" }, { format = "commander", colors = "W" } ]
"""
    (case,) = load_cases(_write(tmp_path, body))
    assert len(case.constraints) == 2
    assert case.constraints[0].color_identity is None
    assert case.constraints[1].color_identity == frozenset({"W"})


def test_empty_colors_means_colorless_not_unconstrained(tmp_path: Path) -> None:
    """The distinction `Constraints` puts in the type must survive parsing."""
    body = MINIMAL.replace('{ format = "commander" }', '{ format = "commander", colors = "" }')
    (case,) = load_cases(_write(tmp_path, body))
    assert case.constraints[0].color_identity == frozenset()


def test_platform_is_carried_when_given(tmp_path: Path) -> None:
    body = MINIMAL.replace(
        '{ format = "commander" }', '{ format = "commander", platform = "arena" }'
    )
    (case,) = load_cases(_write(tmp_path, body))
    assert case.constraints[0].platform == "arena"


# --- structural validation -------------------------------------------------


def test_unknown_predicate_kind_raises(tmp_path: Path) -> None:
    body = MINIMAL.replace('{ keyword = "Connive" }', '{ type_line = "Goblin" }')
    with pytest.raises(UnknownPredicateKindError, match="type_line"):
        load_cases(_write(tmp_path, body))


def test_missing_rationale_raises(tmp_path: Path) -> None:
    body = "\n".join(line for line in MINIMAL.splitlines() if "rationale" not in line)
    with pytest.raises(MalformedCaseError, match="rationale"):
        load_cases(_write(tmp_path, body))


def test_an_unknown_field_raises_rather_than_being_ignored(tmp_path: Path) -> None:
    body = MINIMAL + '\nnote = "typo for rationale"\n'
    with pytest.raises(MalformedCaseError, match="note"):
        load_cases(_write(tmp_path, body))


def test_a_predicate_naming_two_kinds_raises(tmp_path: Path) -> None:
    body = MINIMAL.replace(
        '{ keyword = "Connive" }', '{ keyword = "Connive", oracle_text = "connive" }'
    )
    with pytest.raises(MalformedCaseError, match="exactly one"):
        load_cases(_write(tmp_path, body))


# --- the corpus check ------------------------------------------------------


def test_a_predicate_no_allowed_card_satisfies_raises_naming_the_case(
    tmp_path: Path, corpus: pl.DataFrame
) -> None:
    """A mistyped keyword gives base rate 0 and an undefined lift.

    That is a malformed case, not a result, and it must fail before a run
    spends forty seconds loading a model.
    """
    body = MINIMAL.replace('"Connive"', '"Conniv"')
    cases = load_cases(_write(tmp_path, body))
    with pytest.raises(MalformedCaseError, match="connive"):
        validate_against_corpus(cases, corpus)


def test_a_satisfiable_predicate_passes_the_corpus_check(
    tmp_path: Path, corpus: pl.DataFrame
) -> None:
    validate_against_corpus(load_cases(_write(tmp_path, MINIMAL)), corpus)


# --- the shipped golden set ------------------------------------------------


def test_the_shipped_golden_set_parses() -> None:
    cases = load_cases(golden_path())
    assert cases, "the golden set must not be empty"
    assert len({case.id for case in cases}) == len(cases)


def test_every_shipped_case_carries_a_rationale() -> None:
    assert all(case.rationale.strip() for case in load_cases(golden_path()))
