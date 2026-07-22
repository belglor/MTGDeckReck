"""Tests for the structural `is this a real card` predicate (ADR 0013).

Fixture-driven and run through `normalize_card` + `build_frame` so the predicate
is exercised against the real corpus schema, not a hand-built frame. The last
test runs against `data/cards.parquet` if it is present and is skipped otherwise;
it is the invariant that actually protects the corpus. Nothing here touches the
network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mtg_rag.corpus import is_real, is_real_card, real_cards
from mtg_rag.corpus_config import EXCLUDED_LAYOUTS, EXCLUDED_SET_TYPES
from mtg_rag.ingest.normalize import build_frame, normalize_card

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"
CORPUS = Path("data/cards.parquet")


def _fixtures() -> dict[str, dict[str, Any]]:
    raw = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in raw if line.strip()]
    return {card["name"]: card for card in cards}


@pytest.fixture(scope="module")
def cards() -> dict[str, dict[str, Any]]:
    return _fixtures()


@pytest.fixture(scope="module")
def survivors(cards: dict[str, dict[str, Any]]) -> set[str]:
    """Names of the fixtures that survive the predicate."""
    frame = build_frame([normalize_card(card) for card in cards.values()])
    return set(real_cards(frame)["name"].to_list())


# --- the two shapes of one predicate ---------------------------------------
# `is_real` exists so ingestion can rank a card's printings before a frame
# exists. Two implementations of one rule can drift, so they are pinned to each
# other rather than trusted to stay in step.


def test_scalar_and_frame_predicates_agree() -> None:
    layouts = [None, "normal", "saga", *sorted(EXCLUDED_LAYOUTS)]
    set_types = [None, "core", "expansion", "planechase", *sorted(EXCLUDED_SET_TYPES)]
    cases = [(layout, set_type) for layout in layouts for set_type in set_types]

    frame = pl.DataFrame(
        {"layout": [layout for layout, _ in cases], "set_type": [st for _, st in cases]},
        schema={"layout": pl.String, "set_type": pl.String},
    )
    from_frame = frame.select(is_real_card().alias("real"))["real"].to_list()
    from_scalar = [is_real(layout, set_type) for layout, set_type in cases]

    assert from_scalar == from_frame


def test_scalar_predicate_treats_absent_values_as_real() -> None:
    assert is_real(None, None)


def test_scalar_predicate_rejects_memorabilia() -> None:
    # The 30th Anniversary Edition case that drops Tundra when a memorabilia
    # printing is allowed to represent a real card.
    assert not is_real("normal", "memorabilia")


# --- real cards survive -----------------------------------------------------


def test_normal_card_is_real(survivors: set[str]) -> None:
    assert "Sythis, Harvest's Hand" in survivors


def test_split_card_is_real(survivors: set[str]) -> None:
    assert "Wear // Tear" in survivors


# --- structural non-cards are excluded by layout ----------------------------


def test_art_series_card_is_not_real(survivors: set[str]) -> None:
    assert "Chillerpillar // Chillerpillar" not in survivors


def test_token_and_double_faced_token_are_not_real(survivors: set[str]) -> None:
    assert "Sheep" not in survivors
    assert "Human // Wolf" not in survivors


def test_emblem_is_not_real(survivors: set[str]) -> None:
    assert "Sorin, Lord of Innistrad Emblem" not in survivors


def test_planar_scheme_and_vanguard_layouts_are_not_real(survivors: set[str]) -> None:
    assert "Interplanar Tunnel" not in survivors  # planar
    assert "All in Good Time" not in survivors  # scheme
    assert "Ertai" not in survivors  # vanguard


# --- the traps: where the layout vs set_type line matters -------------------


def test_planechase_set_card_with_normal_layout_is_real(survivors: set[str]) -> None:
    # `Akroma's Vengeance` has set_type == "planechase" — the same set_type the
    # planar object `Interplanar Tunnel` carries — but layout "normal". A
    # set-type rule would destroy 94 real cards; the layout rule keeps them.
    assert "Akroma's Vengeance" in survivors


def test_memorabilia_set_type_is_not_real(survivors: set[str]) -> None:
    # `Proposal` is a Celebration card: layout "normal", set_type "memorabilia".
    # It survives the layout rule and is caught only by the set_type rule.
    assert "Proposal" not in survivors


def test_un_set_card_is_real(survivors: set[str]) -> None:
    # `Steamflogger Boss` is set_type "funny". Silver-border legality is a
    # legality property (ADR 0001), not structure — an Un-card is structurally
    # a card and stays in the index.
    assert "Steamflogger Boss" in survivors


def test_digital_only_card_is_real(survivors: set[str]) -> None:
    # `Angel of Eternal Dawn` is an Alchemy card, games == ["arena"]. Digital-
    # only-ness is what the `games` retrieval filter is for, not this predicate.
    assert "Angel of Eternal Dawn" in survivors


# --- shape and null-safety --------------------------------------------------


def test_is_real_card_returns_a_composable_expression() -> None:
    # The contract channels.py depends on: it composes into `.filter()`.
    assert isinstance(is_real_card(), pl.Expr)


def test_real_cards_matches_the_predicate(cards: dict[str, dict[str, Any]]) -> None:
    frame = build_frame([normalize_card(card) for card in cards.values()])
    assert real_cards(frame).equals(frame.filter(is_real_card()))


def test_card_with_null_set_type_is_real(cards: dict[str, dict[str, Any]]) -> None:
    # A card is dropped only when a value actively matches an exclusion list,
    # never because a field is absent — a null set_type must not filter a card
    # out via null propagation.
    raw = dict(cards["Sol Ring"])
    raw.pop("set_type", None)
    frame = build_frame([normalize_card(raw)])
    assert real_cards(frame).height == 1


# --- the corpus invariant (real parquet only) -------------------------------


def test_predicate_drops_no_commander_legal_card() -> None:
    if not CORPUS.exists():
        pytest.skip(f"no corpus at {CORPUS}; run `just ingest`")
    frame = pl.read_parquet(CORPUS)
    kept = real_cards(frame)

    assert kept.height < frame.height  # the predicate actually removes non-cards
    before = frame.filter(pl.col("legal_commander") == "legal").height
    after = kept.filter(pl.col("legal_commander") == "legal").height
    assert before == after  # ADR 0013: not one commander-legal card is dropped
