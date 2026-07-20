"""Tests for the Scryfall card projection.

Fixtures in `tests/fixtures/cards.jsonl` are real card objects captured from the
Scryfall API. Nothing here touches the network.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mtg_rag.ingest.normalize import MalformedCardError, build_frame, normalize_card

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"


def _fixtures() -> dict[str, dict[str, Any]]:
    raw = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in raw if line.strip()]
    return {card["name"]: card for card in cards}


@pytest.fixture(scope="module")
def cards() -> dict[str, dict[str, Any]]:
    return _fixtures()


# --- the projection -------------------------------------------------------


def test_normal_card_projects_expected_fields(cards: dict[str, dict[str, Any]]) -> None:
    record = normalize_card(cards["Sythis, Harvest's Hand"])

    assert record.oracle_id == "0fc64fd6-f057-4056-9dca-47accb7ff036"
    assert record.name == "Sythis, Harvest's Hand"
    assert record.oracle_text is not None
    assert "you gain 1 life and draw a card" in record.oracle_text
    assert record.type_line is not None
    assert "Nymph" in record.type_line
    assert record.color_identity == ["G", "W"]
    assert record.cmc == 2.0
    assert record.power == "1"
    assert record.toughness == "2"
    assert record.layout == "normal"
    assert record.set_code == "cmm"
    assert record.set_type == "masters"
    assert record.legalities["commander"] == "legal"


def test_flavor_text_is_captured_when_present(cards: dict[str, dict[str, Any]]) -> None:
    record = normalize_card(cards["Sythis, Harvest's Hand"])
    assert record.flavor_text is not None
    assert "Karametra" in record.flavor_text


def test_missing_flavor_text_is_none_not_empty_string(cards: dict[str, dict[str, Any]]) -> None:
    record = normalize_card(cards["Black Lotus"])
    assert record.flavor_text is None


# --- multi-face cards: ADR 0002 says one card is one record ---------------


def test_split_card_is_one_record_carrying_both_halves(
    cards: dict[str, dict[str, Any]],
) -> None:
    record = normalize_card(cards["Wear // Tear"])

    assert record.oracle_text is not None
    assert "Destroy target artifact" in record.oracle_text
    assert "Destroy target enchantment" in record.oracle_text
    assert record.name == "Wear // Tear"
    assert record.type_line == "Instant // Instant"


def test_modal_dfc_joins_both_faces_flavor_text(cards: dict[str, dict[str, Any]]) -> None:
    record = normalize_card(cards["Agadeem's Awakening // Agadeem, the Undercrypt"])

    assert record.layout == "modal_dfc"
    assert record.flavor_text is not None
    assert "death-hour" in record.flavor_text
    assert "entombed" in record.flavor_text
    assert record.oracle_text is not None
    assert record.oracle_text.count("\n//\n") == 1


def test_face_mana_costs_skip_empty_faces(cards: dict[str, dict[str, Any]]) -> None:
    # The land face of a modal DFC has mana_cost "", which must not produce a
    # dangling separator.
    record = normalize_card(cards["Agadeem's Awakening // Agadeem, the Undercrypt"])
    assert record.mana_cost == "{X}{B}{B}{B}"


# --- preprocessing: lowering + Unicode normalization -----------------------


def test_legalities_keys_and_values_are_lowercased(cards: dict[str, dict[str, Any]]) -> None:
    # ADR 0001 filters on legality; a casing mismatch there would silently
    # drop a card rather than raise, so this is enforced rather than trusted.
    raw = dict(cards["Sol Ring"])
    raw["legalities"] = {"Commander": "Legal", "vintage": "RESTRICTED"}

    record = normalize_card(raw)

    assert record.legalities == {"commander": "legal", "vintage": "restricted"}


def test_categorical_fields_are_lowercased(cards: dict[str, dict[str, Any]]) -> None:
    raw = dict(cards["Sol Ring"], layout="Normal", rarity="Uncommon", set="MSC", set_type="Commander")
    raw["games"] = ["Paper", "MTGO"]

    record = normalize_card(raw)

    assert record.layout == "normal"
    assert record.rarity == "uncommon"
    assert record.set_code == "msc"
    assert record.set_type == "commander"
    assert record.games == ["paper", "mtgo"]


def test_display_text_fields_keep_their_case(cards: dict[str, dict[str, Any]]) -> None:
    # Lowering is for exact-match filter fields only — name/oracle text/type
    # line are read by people and embedded, so their casing is meaningful.
    record = normalize_card(cards["Sol Ring"])

    assert record.name == "Sol Ring"
    assert record.type_line == "Artifact"


def test_text_fields_are_unicode_nfc_normalized(cards: dict[str, dict[str, Any]]) -> None:
    # Scryfall's export happens to ship NFC-composed text, but nothing
    # guarantees it — a decomposed form of the same name would otherwise
    # embed differently from the composed one for no visible reason.
    decomposed_name = unicodedata.normalize("NFD", "Sol Ring é")  # e + combining acute
    raw = dict(cards["Sol Ring"], name=decomposed_name)
    assert not unicodedata.is_normalized("NFC", decomposed_name)

    record = normalize_card(raw)

    assert record.name == unicodedata.normalize("NFC", decomposed_name)
    assert unicodedata.is_normalized("NFC", record.name)


# --- edge cases that bite downstream --------------------------------------


def test_colorless_card_has_empty_color_identity(cards: dict[str, dict[str, Any]]) -> None:
    # Empty color identity is the case that breaks array-valued metadata in a
    # vector store, so pin the shape here.
    record = normalize_card(cards["Sol Ring"])
    assert record.color_identity == []


def test_absent_optional_numerics_are_none(cards: dict[str, dict[str, Any]]) -> None:
    record = normalize_card(cards["Black Lotus"])
    assert record.edhrec_rank is None
    assert record.price_usd is None
    assert record.price_tix == 47.23


def test_missing_oracle_id_raises(cards: dict[str, dict[str, Any]]) -> None:
    broken = dict(cards["Sol Ring"])
    del broken["oracle_id"]
    with pytest.raises(MalformedCardError):
        normalize_card(broken)


def test_blank_oracle_id_raises(cards: dict[str, dict[str, Any]]) -> None:
    broken = dict(cards["Sol Ring"], oracle_id="")
    with pytest.raises(MalformedCardError):
        normalize_card(broken)


# --- frame assembly --------------------------------------------------------


def test_legality_flattening_preserves_banned_and_restricted(
    cards: dict[str, dict[str, Any]],
) -> None:
    frame = build_frame([normalize_card(c) for c in cards.values()])
    lotus = frame.filter(pl.col("name") == "Black Lotus")

    assert lotus["legal_commander"].item() == "banned"
    assert lotus["legal_vintage"].item() == "restricted"
    assert lotus["legal_modern"].item() == "not_legal"


def test_every_format_becomes_its_own_column(cards: dict[str, dict[str, Any]]) -> None:
    frame = build_frame([normalize_card(c) for c in cards.values()])
    legal_columns = [c for c in frame.columns if c.startswith("legal_")]

    expected = len(cards["Sol Ring"]["legalities"])
    assert len(legal_columns) == expected
    assert "legalities" not in frame.columns


def test_oracle_id_is_unique_across_the_corpus(cards: dict[str, dict[str, Any]]) -> None:
    frame = build_frame([normalize_card(c) for c in cards.values()])
    assert frame["oracle_id"].n_unique() == frame.height


def test_build_frame_rejects_duplicate_oracle_ids(cards: dict[str, dict[str, Any]]) -> None:
    record = normalize_card(cards["Sol Ring"])
    with pytest.raises(ValueError, match="oracle_id"):
        build_frame([record, record])


def test_build_frame_on_empty_input_raises(cards: dict[str, dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="no cards"):
        build_frame([])


def test_released_at_is_a_date_column(cards: dict[str, dict[str, Any]]) -> None:
    frame = build_frame([normalize_card(c) for c in cards.values()])
    assert frame.schema["released_at"] == pl.Date
