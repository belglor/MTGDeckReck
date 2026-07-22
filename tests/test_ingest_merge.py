"""Tests for collapsing many printings into one record per card.

`default_cards` ships one object per printing, so ingestion sees a card once per
time it was printed. These cover the two rules that decide what survives: a
representative printing supplies every single-valued field, and flavor text is
the documented exception.

`tests/fixtures/printings.jsonl` holds three real Sol Ring printings captured
from Scryfall — a 2010 one with flavor text, a 2026-07-20 one with different
flavor text, and the 2026-07-27 reprint that has none. Sol Ring is the extreme
case in the live corpus: 54 distinct flavor texts across its printings, and a
newest printing carrying none of them. Nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mtg_rag.ingest.merge import merge_printings
from mtg_rag.ingest.normalize import CardRecord, normalize_card

PRINTINGS = Path(__file__).parent / "fixtures" / "printings.jsonl"


@pytest.fixture(scope="module")
def sol_ring_printings() -> list[CardRecord]:
    """Three real Sol Ring printings, oldest first."""
    raw = PRINTINGS.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in raw if line.strip()]
    return [normalize_card(card) for card in cards]


def _printing(**overrides: Any) -> CardRecord:
    """A CardRecord with plausible defaults, for the fields a test cares about.

    Built directly rather than through `normalize_card`, because merging takes
    records as its input — routing through the projection would test that
    instead, and the fixture-driven tests below already cover the real shape.
    """
    fields: dict[str, Any] = {
        "oracle_id": "0fc64fd6-f057-4056-9dca-47accb7ff036",
        "name": "Sol Ring",
        "oracle_text": "{T}: Add {C}{C}.",
        "flavor_text": None,
        "type_line": "Artifact",
        "mana_cost": "{1}",
        "cmc": 1.0,
        "colors": [],
        "color_identity": [],
        "keywords": [],
        "power": None,
        "toughness": None,
        "loyalty": None,
        "layout": "normal",
        "rarity": "uncommon",
        "set_code": "lea",
        "set_name": "Limited Edition Alpha",
        "set_type": "core",
        "released_at": "1993-08-05",
        "platforms": ["paper"],
        "reserved": False,
        "edhrec_rank": 1,
        "price_usd": 1.0,
        "price_eur": None,
        "price_tix": None,
        "legalities": {"commander": "legal"},
    }
    fields.update(overrides)
    return CardRecord(**fields)


# --- one record per card ---------------------------------------------------


def test_printings_of_one_card_collapse_to_one_record() -> None:
    merged = merge_printings([_printing(set_code="lea"), _printing(set_code="cmm")])
    assert len(merged) == 1


def test_several_cards_are_grouped_independently() -> None:
    merged = merge_printings(
        [
            _printing(oracle_id="aaa", set_code="lea"),
            _printing(oracle_id="bbb", set_code="lea"),
            _printing(oracle_id="aaa", set_code="cmm"),
        ]
    )
    assert {record.oracle_id for record in merged} == {"aaa", "bbb"}
    assert len(merged) == 2


def test_merging_nothing_returns_nothing() -> None:
    assert merge_printings([]) == []


def test_single_printing_card_passes_through_unchanged() -> None:
    only = _printing(flavor_text="Alone.")
    assert merge_printings([only]) == [only]


def test_output_is_ordered_by_oracle_id() -> None:
    # Row order should not depend on the order Scryfall happened to ship
    # printings in, so two runs over the same data produce the same parquet.
    merged = merge_printings(
        [_printing(oracle_id="ccc"), _printing(oracle_id="aaa"), _printing(oracle_id="bbb")]
    )
    assert [record.oracle_id for record in merged] == ["aaa", "bbb", "ccc"]


# --- the representative printing -------------------------------------------


def test_representative_printing_is_the_most_recent() -> None:
    merged = merge_printings(
        [
            _printing(set_code="lea", released_at="1993-08-05"),
            _printing(set_code="cmm", released_at="2023-08-04"),
            _printing(set_code="v10", released_at="2010-08-27"),
        ]
    )
    assert merged[0].set_code == "cmm"
    assert merged[0].released_at == "2023-08-04"


def test_representative_printing_supplies_rarity_set_and_prices_together() -> None:
    # The point of one representative printing is that a row describes a single
    # physical card: its price is the price of the set and rarity beside it,
    # not a value stitched together from three different printings.
    merged = merge_printings(
        [
            _printing(
                set_code="lea",
                set_name="Limited Edition Alpha",
                released_at="1993-08-05",
                rarity="uncommon",
                price_usd=3000.0,
            ),
            _printing(
                set_code="cmm",
                set_name="Commander Masters",
                released_at="2023-08-04",
                rarity="mythic",
                price_usd=2.5,
            ),
        ]
    )
    record = merged[0]

    assert record.set_code == "cmm"
    assert record.set_name == "Commander Masters"
    assert record.rarity == "mythic"
    assert record.price_usd == 2.5


def test_ties_on_release_date_break_deterministically() -> None:
    same_day = "2023-08-04"
    forwards = merge_printings(
        [
            _printing(set_code="aaa", released_at=same_day),
            _printing(set_code="zzz", released_at=same_day),
        ]
    )
    backwards = merge_printings(
        [
            _printing(set_code="zzz", released_at=same_day),
            _printing(set_code="aaa", released_at=same_day),
        ]
    )
    assert forwards[0].set_code == backwards[0].set_code


def test_missing_released_at_does_not_crash_the_ordering() -> None:
    merged = merge_printings(
        [
            _printing(set_code="dated", released_at="2020-01-01"),
            _printing(set_code="undated", released_at=None),
        ]
    )
    # A printing with no date cannot be shown to be the most recent, so it does
    # not win — but it must not raise on the way to that conclusion either.
    assert merged[0].set_code == "dated"


def test_a_card_with_no_dated_printing_still_merges() -> None:
    merged = merge_printings(
        [_printing(set_code="aaa", released_at=None), _printing(set_code="bbb", released_at=None)]
    )
    assert len(merged) == 1


def test_a_memorabilia_printing_never_represents_a_real_card() -> None:
    # The Tundra case ([ADR 0016]): a memorabilia reprint as representative
    # hands a real card a non-card's set type, and the predicate then discards
    # the card itself.
    merged = merge_printings(
        [
            _printing(set_code="lea", set_type="core", released_at="1993-08-05"),
            _printing(set_code="30a", set_type="memorabilia", released_at="2022-11-28"),
        ]
    )

    assert merged[0].set_code == "lea"
    assert merged[0].set_type == "core"


def test_a_token_printing_never_represents_a_real_card() -> None:
    merged = merge_printings(
        [
            _printing(set_code="cmm", layout="normal", released_at="2023-08-04"),
            _printing(set_code="tcmm", layout="token", released_at="2024-01-01"),
        ]
    )
    assert merged[0].layout == "normal"


def test_a_card_with_only_excluded_printings_still_merges() -> None:
    # A genuine token has no real printing to prefer. It must still collapse to
    # one record — the structural predicate excludes it downstream, which is
    # where that decision belongs.
    merged = merge_printings(
        [
            _printing(set_code="tclb", layout="token", released_at="2022-06-10"),
            _printing(set_code="tltr", layout="token", released_at="2023-06-23"),
        ]
    )

    assert len(merged) == 1
    assert merged[0].set_code == "tltr"


def test_invariant_fields_come_through_unchanged() -> None:
    # Oracle text, type line and legality are oracle-level: they are identical
    # on every printing, measured across all 116,138 of them. Taking them from
    # the representative printing is therefore arbitrary and safe — this pins
    # that, so a future change to the rule cannot quietly alter them.
    merged = merge_printings(
        [
            _printing(set_code="lea", released_at="1993-08-05"),
            _printing(set_code="cmm", released_at="2023-08-04"),
        ]
    )
    record = merged[0]

    assert record.oracle_text == "{T}: Add {C}{C}."
    assert record.type_line == "Artifact"
    assert record.legalities == {"commander": "legal"}
    assert record.name == "Sol Ring"


# --- platforms: unioned across every printing ------------------------------


def test_platforms_union_across_printings() -> None:
    # The Palinchron case in miniature: an MTGO-only reprint and a paper
    # original. The card is playable on both, and neither printing says so
    # alone.
    merged = merge_printings(
        [
            _printing(set_code="ulg", released_at="1999-02-15", platforms=["paper", "mtgo"]),
            _printing(set_code="vma", released_at="2014-06-16", platforms=["mtgo"]),
        ]
    )
    assert merged[0].platforms == ["mtgo", "paper"]


def test_platforms_ignore_the_representative_printing_rule() -> None:
    # Every other single-valued field comes from the representative printing.
    # Platforms do not: the union is the whole point, so a narrower newest
    # printing must not shrink it.
    merged = merge_printings(
        [
            _printing(set_code="lea", released_at="1993-08-05", platforms=["paper"]),
            _printing(set_code="mkm", released_at="2024-02-09", platforms=["arena"]),
        ]
    )
    assert merged[0].set_code == "mkm"
    assert merged[0].platforms == ["arena", "paper"]


def test_single_printing_card_keeps_its_own_platforms() -> None:
    merged = merge_printings([_printing(platforms=["paper", "arena"])])
    assert merged[0].platforms == ["arena", "paper"]


def test_card_with_only_unknown_media_gets_empty_platforms() -> None:
    # The ~18 astral / sega curiosities. They keep their row — excluding them
    # is the platform filter's job, not ingestion's.
    merged = merge_printings([_printing(platforms=[])])
    assert len(merged) == 1
    assert merged[0].platforms == []


def test_platforms_are_deduplicated_and_ordered() -> None:
    merged = merge_printings(
        [
            _printing(set_code="a", released_at="2000-01-01", platforms=["paper", "mtgo"]),
            _printing(set_code="b", released_at="2001-01-01", platforms=["mtgo", "paper"]),
        ]
    )
    assert merged[0].platforms == ["mtgo", "paper"]


# --- flavor text: the documented exception ---------------------------------


def test_flavor_text_comes_from_the_latest_printing_that_has_one() -> None:
    merged = merge_printings(
        [
            _printing(released_at="1993-08-05", set_code="lea", flavor_text="Oldest."),
            _printing(released_at="2010-08-27", set_code="v10", flavor_text="Middle."),
            _printing(released_at="2023-08-04", set_code="cmm", flavor_text=None),
        ]
    )
    assert merged[0].flavor_text == "Middle."


def test_card_keeps_flavor_when_the_newest_printing_has_none(
    sol_ring_printings: list[CardRecord],
) -> None:
    # The 1,804-card case, on real data. Sol Ring's newest printing carries no
    # flavor text; taking flavor from the representative printing would drop it
    # from the flavor channel entirely.
    newest = max(sol_ring_printings, key=lambda record: record.released_at or "")
    assert newest.flavor_text is None

    merged = merge_printings(sol_ring_printings)

    assert len(merged) == 1
    assert merged[0].flavor_text is not None
    assert "align the aether" in merged[0].flavor_text


def test_flavor_donor_does_not_drag_its_other_fields_along(
    sol_ring_printings: list[CardRecord],
) -> None:
    # Only the text crosses over. Everything else still describes the
    # representative printing.
    merged = merge_printings(sol_ring_printings)
    record = merged[0]

    assert record.released_at == "2026-07-27"
    assert record.rarity == "rare"


def test_card_with_no_flavor_anywhere_stays_null() -> None:
    merged = merge_printings(
        [_printing(flavor_text=None), _printing(set_code="cmm", flavor_text=None)]
    )
    assert merged[0].flavor_text is None


def test_whitespace_is_not_mistaken_for_flavor_text() -> None:
    # `normalize_card` already maps empty strings to None; this guards the
    # merge against treating a whitespace-only value as a real donor.
    merged = merge_printings(
        [
            _printing(released_at="1993-08-05", set_code="lea", flavor_text="Real flavor."),
            _printing(released_at="2023-08-04", set_code="cmm", flavor_text="   "),
        ]
    )
    assert merged[0].flavor_text == "Real flavor."
