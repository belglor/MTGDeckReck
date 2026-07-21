"""Tests for per-channel text composition ([ADR 0007], [ADR 0014]).

Fixture-driven through `normalize_card` + `build_frame`, so the channels run
against the real corpus schema. Pure and offline — no model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from mtg_rag.embed.channels import channel_expr, channel_frame
from mtg_rag.embed.config import CHANNELS, TEXT_COLUMN, Channel
from mtg_rag.ingest.normalize import build_frame, normalize_card

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"


def _fixtures() -> dict[str, dict[str, Any]]:
    raw = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in raw if line.strip()]
    return {card["name"]: card for card in cards}


@pytest.fixture(scope="module")
def cards() -> dict[str, dict[str, Any]]:
    return _fixtures()


@pytest.fixture(scope="module")
def frame(cards: dict[str, dict[str, Any]]) -> pl.DataFrame:
    return build_frame([normalize_card(card) for card in cards.values()])


def _texts(frame: pl.DataFrame, channel: Channel) -> dict[str, str]:
    """A channel's text keyed by card name, for readable assertions."""
    composed = channel_frame(frame, channel).join(
        frame.select("oracle_id", "name"), on="oracle_id", how="left"
    )
    names: list[str] = composed["name"].to_list()
    values: list[str] = composed[TEXT_COLUMN].to_list()
    return dict(zip(names, values, strict=True))


def _single(card: dict[str, Any]) -> pl.DataFrame:
    """A one-row corpus frame, for cases no real fixture covers."""
    return build_frame([normalize_card(card)])


# --- what each channel embeds ----------------------------------------------


def test_oracle_channel_folds_in_card_name(frame: pl.DataFrame) -> None:
    # ADR 0007: the name is a rules-text-adjacent identifier, so it rides along
    # with oracle text — and only there.
    text = _texts(frame, "oracle")["Sythis, Harvest's Hand"]

    assert text.startswith("Sythis, Harvest's Hand\n")
    assert "you gain 1 life and draw a card" in text


def test_flavor_channel_is_flavor_text_alone(frame: pl.DataFrame) -> None:
    text = _texts(frame, "flavor")["Sythis, Harvest's Hand"]
    expected = frame.filter(pl.col("name") == "Sythis, Harvest's Hand")["flavor_text"].item()

    assert text == expected
    assert not text.startswith("Sythis, Harvest's Hand")  # no name fold-in


def test_type_channel_is_type_line_alone(frame: pl.DataFrame) -> None:
    # Pins the "no name in the type channel" decision: the fold-in argument is
    # about rules-adjacent prose and does not extend to a controlled vocabulary.
    assert _texts(frame, "type")["Sol Ring"] == "Artifact"


# --- an absent channel produces no row, never a blank one (ADR 0014) --------


def test_card_without_flavor_text_gets_no_flavor_row(frame: pl.DataFrame) -> None:
    assert "Black Lotus" not in _texts(frame, "flavor")
    # ...but it is still reachable through the channels it does populate.
    assert "Black Lotus" in _texts(frame, "oracle")
    assert "Black Lotus" in _texts(frame, "type")


def test_vanilla_card_without_oracle_text_gets_no_oracle_row(
    cards: dict[str, dict[str, Any]],
) -> None:
    # No real fixture is vanilla, so strip the oracle text off one. A bare name
    # must not be embedded as if it were rules text.
    vanilla = dict(cards["Llanowar Elves"])
    vanilla.pop("oracle_text", None)
    frame = _single(vanilla)

    assert channel_frame(frame, "oracle").height == 0
    assert channel_frame(frame, "type").height == 1


def test_whitespace_only_text_is_treated_as_absent(cards: dict[str, dict[str, Any]]) -> None:
    blank = dict(cards["Sol Ring"], flavor_text="   \n  ")
    assert channel_frame(_single(blank), "flavor").height == 0


# --- shape ------------------------------------------------------------------


def test_face_separator_survives_into_channel_text(frame: pl.DataFrame) -> None:
    # A split card is one record carrying both halves (ADR 0002); the separators
    # are meaningful text and must reach the encoder untouched.
    assert "\n//\n" in _texts(frame, "oracle")["Wear // Tear"]
    assert _texts(frame, "type")["Wear // Tear"] == "Instant // Instant"


def test_channel_frame_is_keyed_by_unique_oracle_id(frame: pl.DataFrame) -> None:
    for channel in CHANNELS:
        composed = channel_frame(frame, channel)
        assert composed.columns == ["oracle_id", TEXT_COLUMN]
        assert composed["oracle_id"].n_unique() == composed.height


def test_channel_frame_excludes_cards_the_corpus_predicate_rejects(
    frame: pl.DataFrame,
) -> None:
    # Ertai (vanguard) and Proposal (memorabilia) both carry oracle text, so they
    # would appear here if the predicate were not applied.
    for channel in CHANNELS:
        embedded = set(_texts(frame, channel))
        assert "Ertai" not in embedded
        assert "Proposal" not in embedded
        assert "Sheep" not in embedded
    assert "Sol Ring" in _texts(frame, "oracle")  # the frame is not simply empty


def test_unknown_channel_raises() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        channel_expr(cast("Channel", "rulings"))
