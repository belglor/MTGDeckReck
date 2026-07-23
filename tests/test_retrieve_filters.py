"""Tests for the hard pre-filter: constraints → `oracle_id` allowlist.

Fixture-driven through `normalize_card` + `build_frame`, so the schema under
test is the one ingestion actually writes — legality flattened to `legal_<format>`
columns, `platforms` as a list, `color_identity` uppercase WUBRG. A few edge
cases the fixtures can't express (a legal token, a null legality) build a small
frame directly. Nothing here touches the network, a model, or Chroma.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.ingest.normalize import build_frame, normalize_card
from mtg_rag.retrieve.config import DEFAULT_PLATFORM
from mtg_rag.retrieve.filters import (
    Constraints,
    allowed_ids,
    available_formats,
    color_identity_expr,
    constraint_expr,
    legality_expr,
    platform_expr,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cards.jsonl"
CORPUS = Path("data/cards.parquet")


def _raw() -> dict[str, dict[str, Any]]:
    lines = FIXTURES.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = [json.loads(line) for line in lines if line.strip()]
    return {card["name"]: card for card in cards}


@pytest.fixture(scope="module")
def raw() -> dict[str, dict[str, Any]]:
    return _raw()


@pytest.fixture(scope="module")
def corpus(raw: dict[str, dict[str, Any]]) -> pl.DataFrame:
    return build_frame([normalize_card(card) for card in raw.values()])


def _names_allowed(corpus: pl.DataFrame, constraints: Constraints) -> set[str]:
    ids = set(allowed_ids(corpus, constraints))
    return set(corpus.filter(pl.col(ID_COLUMN).is_in(ids))["name"].to_list())


# --- legality --------------------------------------------------------------


def test_legal_card_is_allowed(corpus: pl.DataFrame) -> None:
    allowed = _names_allowed(corpus, Constraints("commander"))
    assert "Sythis, Harvest's Hand" in allowed


def test_banned_card_is_excluded(corpus: pl.DataFrame) -> None:
    # Black Lotus is banned in commander (and only on mtgo, but legality alone
    # is enough to drop it).
    allowed = _names_allowed(corpus, Constraints("commander", platform="mtgo"))
    assert "Black Lotus" not in allowed


def test_not_legal_card_is_excluded(corpus: pl.DataFrame) -> None:
    allowed = _names_allowed(corpus, Constraints("commander"))
    assert "Chillerpillar // Chillerpillar" not in allowed


def test_restricted_card_is_allowed_in_vintage(corpus: pl.DataFrame) -> None:
    # `restricted` caps you at one copy — a deckbuilding rule, not a ban.
    allowed = _names_allowed(corpus, Constraints("vintage"))
    assert "Sol Ring" in allowed


def test_unknown_format_raises_listing_available_formats(corpus: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="commander") as exc:
        allowed_ids(corpus, Constraints("pauperEDH"))
    # the message names what was available, so a UI typo is diagnosable
    assert "pauperEDH" in str(exc.value)


def test_available_formats_are_the_legal_columns(corpus: pl.DataFrame) -> None:
    formats = available_formats(corpus)
    assert "commander" in formats
    assert "vintage" in formats
    assert all(not fmt.startswith("legal_") for fmt in formats)


def test_null_legality_is_excluded() -> None:
    # A format one card declares and another omits leaves the omitter's
    # `legal_<format>` null. Null is not-playable, guarded rather than left to
    # polars' three-valued `filter` (which would drop it for the wrong reason).
    raw = _raw()
    keeps = dict(raw["Sythis, Harvest's Hand"], legalities={"commander": "legal"})
    drops = dict(raw["Sol Ring"], legalities={"vintage": "legal"})  # no commander key
    frame = build_frame([normalize_card(keeps), normalize_card(drops)])

    allowed = _names_allowed(frame, Constraints("commander"))

    assert "Sythis, Harvest's Hand" in allowed
    assert "Sol Ring" not in allowed


# --- color identity --------------------------------------------------------


def test_card_within_identity_is_allowed(corpus: pl.DataFrame) -> None:
    allowed = _names_allowed(corpus, Constraints("commander", frozenset({"G", "W"})))
    assert "Sythis, Harvest's Hand" in allowed  # GW


def test_card_outside_identity_is_excluded(corpus: pl.DataFrame) -> None:
    allowed = _names_allowed(corpus, Constraints("commander", frozenset({"G"})))
    assert "Sythis, Harvest's Hand" not in allowed  # GW ⊄ {G}


def test_colorless_card_passes_every_identity(corpus: pl.DataFrame) -> None:
    identities: tuple[frozenset[str], ...] = (
        frozenset(),
        frozenset({"W"}),
        frozenset({"G", "W"}),
    )
    for identity in identities:
        allowed = _names_allowed(corpus, Constraints("commander", identity))
        assert "Sol Ring" in allowed, identity  # colorless, commander-legal


def test_none_identity_is_unconstrained(corpus: pl.DataFrame) -> None:
    # None ≠ empty set: None means "don't filter on color at all".
    allowed = _names_allowed(corpus, Constraints("commander", color_identity=None))
    assert "Sythis, Harvest's Hand" in allowed  # GW survives an absent constraint


def test_empty_identity_allows_only_colorless(corpus: pl.DataFrame) -> None:
    allowed = _names_allowed(corpus, Constraints("commander", frozenset()))
    assert "Sol Ring" in allowed  # colorless
    assert "Sythis, Harvest's Hand" not in allowed  # any color is too much


def test_identity_input_is_case_insensitive(corpus: pl.DataFrame) -> None:
    lower = _names_allowed(corpus, Constraints("commander", frozenset({"g", "w"})))
    upper = _names_allowed(corpus, Constraints("commander", frozenset({"G", "W"})))
    assert "Sythis, Harvest's Hand" in lower
    assert lower == upper


# --- platform --------------------------------------------------------------


def test_paper_is_the_default_platform() -> None:
    assert Constraints("commander").platform == DEFAULT_PLATFORM == "paper"


def test_alchemy_card_is_excluded_under_paper(corpus: pl.DataFrame) -> None:
    # `Angel of Eternal Dawn` is arena-only. Under paper it cannot appear,
    # whatever its legality.
    kept = corpus.filter(platform_expr("paper"))["name"].to_list()
    assert "Angel of Eternal Dawn" not in kept


def test_alchemy_card_is_included_under_arena(corpus: pl.DataFrame) -> None:
    kept = corpus.filter(platform_expr("arena"))["name"].to_list()
    assert "Angel of Eternal Dawn" in kept


def test_paper_only_card_is_excluded_under_arena(corpus: pl.DataFrame) -> None:
    # `Steamflogger Boss` is paper-only and commander-legal, so only the
    # platform dimension can drop it.
    paper = _names_allowed(corpus, Constraints("commander", platform="paper"))
    arena = _names_allowed(corpus, Constraints("commander", platform="arena"))
    assert "Steamflogger Boss" in paper
    assert "Steamflogger Boss" not in arena


def test_unknown_platform_raises() -> None:
    with pytest.raises(ValueError, match="mtga"):
        platform_expr("mtga")  # it's "arena", a UI typo should say so


# --- composition -----------------------------------------------------------


def test_structural_non_card_is_excluded_even_when_legal(
    raw: dict[str, dict[str, Any]],
) -> None:
    # A token with legal_commander == "legal" must still be dropped: the
    # structural predicate is not a legality question ([ADR 0013]).
    token = dict(raw["Sythis, Harvest's Hand"], layout="token")
    frame = build_frame([normalize_card(token)])
    assert frame["legal_commander"].item() == "legal"

    assert allowed_ids(frame, Constraints("commander")) == []


def test_allowed_ids_are_sorted_and_unique(corpus: pl.DataFrame) -> None:
    ids = allowed_ids(corpus, Constraints("commander", color_identity=None))
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_constraints_compose_into_a_single_filter_pass(corpus: pl.DataFrame) -> None:
    constraints = Constraints("commander", frozenset({"G", "W"}), platform="paper")
    from_helper = set(allowed_ids(corpus, constraints))
    from_expr = set(corpus.filter(constraint_expr(constraints, corpus))[ID_COLUMN].to_list())
    assert from_helper == from_expr


def test_no_card_survives_contradictory_constraints(raw: dict[str, dict[str, Any]]) -> None:
    # An empty allowlist is a valid answer, not an error. Angel of Eternal Dawn
    # is arena-only and not commander-legal, so commander+paper matches nothing.
    frame = build_frame([normalize_card(raw["Angel of Eternal Dawn"])])
    assert allowed_ids(frame, Constraints("commander", platform="paper")) == []


def test_legality_expr_rejects_an_unknown_format(corpus: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="nope"):
        legality_expr("nope", corpus)


def test_color_identity_expr_is_pure(corpus: pl.DataFrame) -> None:
    # Returns a composable expression, like the other predicates.
    assert isinstance(color_identity_expr(frozenset({"G"})), pl.Expr)


# --- the real corpus (skipped when absent) ---------------------------------
# Following ADR-driven practice and issue #41's lesson: assert relationships
# and loose bounds, never frozen absolute counts, which drift every snapshot.


def _real_corpus() -> pl.DataFrame | None:
    return pl.read_parquet(CORPUS) if CORPUS.exists() else None


def test_commander_allowlist_is_plausible() -> None:
    frame = _real_corpus()
    if frame is None:
        pytest.skip(f"no corpus at {CORPUS}; run `just ingest`")
    ids = allowed_ids(frame, Constraints("commander", color_identity=None))
    # Tens of thousands of commander-legal paper cards, well short of the corpus.
    assert 25_000 < len(ids) < frame.height


def test_allowlist_narrows_monotonically_with_color() -> None:
    frame = _real_corpus()
    if frame is None:
        pytest.skip(f"no corpus at {CORPUS}; run `just ingest`")
    everything = set(allowed_ids(frame, Constraints("commander", color_identity=None)))
    gw = set(allowed_ids(frame, Constraints("commander", frozenset({"G", "W"}))))
    colorless = set(allowed_ids(frame, Constraints("commander", frozenset())))

    assert colorless < gw < everything  # strict-subset chain


def test_every_allowed_id_is_in_the_corpus() -> None:
    frame = _real_corpus()
    if frame is None:
        pytest.skip(f"no corpus at {CORPUS}; run `just ingest`")
    ids = allowed_ids(frame, Constraints("commander", frozenset({"B"})))
    corpus_ids = set(frame[ID_COLUMN].to_list())
    assert set(ids) <= corpus_ids
