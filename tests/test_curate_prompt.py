"""Tests for assembling the curation prompt from a request, a template, and a pool.

`build_messages` is curation's pure step: the whole format template ([ADR 0025]),
the user's theme, and the retrieved candidates become the `system`/`user` strings
`LLMClient.complete` takes ([ADR 0021]). It asks for the flat
`[{oracle_id, role, rationale}]` recommendation ([ADR 0024]) grouped by a
model-authored role with a checkable argument ([ADR 0005]), and it never mentions
legality, colour identity, or platform ([ADR 0001]) nor any ranking number
([ADR 0008]).

Nothing here touches a model or the filesystem: the template arrives as a string
and the cards as data, so these tests pass both directly.
"""

from __future__ import annotations

from mtg_rag.curate.prompt import CurationCard, build_messages

# A realistic theme and a stand-in template. The template text is a sentinel so
# a test can assert the function embeds *the argument it was handed*, not a file
# it read itself — the read lives in the curation entry point.
_REQUEST = "a spooky graveyard deck that mills itself"
_TEMPLATE = "## Heuristics\n- Ramp: ~10\n- Card draw: ~10\n- Theme cards: the rest"

_LOAM = CurationCard(
    oracle_id="abc123",
    name="Life from the Loam",
    mana_cost="{1}{B}{G}",
    type_line="Sorcery",
    oracle_text="Return up to three target land cards from your graveyard to your hand.",
    flavor_text="The soil remembers every seed it swallowed.",
    purposes=("theme payoff", "recursion"),
)
_CRYPT = CurationCard(
    oracle_id="def456",
    name="Sunken Crypt",
    mana_cost="",
    type_line="Land",
    oracle_text="",
    flavor_text="",
    purposes=("mana base",),
)
_CARDS = (_LOAM, _CRYPT)


def test_the_template_text_is_embedded_verbatim() -> None:
    # Proves purity: the function consumes the passed string, so it can't be
    # reaching for a file of its own.
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    assert _TEMPLATE in messages.system


def test_the_request_text_is_embedded_verbatim() -> None:
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    assert _REQUEST in messages.user


def test_every_cards_name_and_purposes_reach_the_prompt() -> None:
    # The purpose is the starting hypothesis for a card's role ([ADR 0005]) —
    # it is the reason retrieval found this card, and the model can't weigh it
    # if it never sees it.
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    for card in _CARDS:
        assert card.name in messages.user
        for purpose in card.purposes:
            assert purpose in messages.user


def test_a_cards_text_reaches_the_prompt() -> None:
    # Theme fit is argued from the card's own words, so all of them travel.
    messages = build_messages(_REQUEST, _TEMPLATE, (_LOAM,))

    for field in (_LOAM.mana_cost, _LOAM.type_line, _LOAM.oracle_text, _LOAM.flavor_text):
        assert field in messages.user


def test_a_cards_oracle_id_reaches_the_prompt() -> None:
    # The model returns `oracle_id`s and the parser checks each one against the
    # pool ([ADR 0024]), so the ids have to be in front of the model to copy —
    # otherwise every reply is an invented id and every run raises.
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    for card in _CARDS:
        assert card.oracle_id in messages.user


def test_a_card_missing_optional_text_leaves_no_empty_label() -> None:
    # Lands have no mana cost and plenty of cards have no flavour text. An
    # empty value renders as nothing at all rather than a dangling label the
    # model could read as a card with a blank type.
    messages = build_messages(_REQUEST, _TEMPLATE, (_CRYPT,))

    assert "mana cost:" not in messages.user
    assert "flavor:" not in messages.user
    assert _CRYPT.type_line in messages.user


def test_the_output_schema_is_specified() -> None:
    # The model is told the exact shape to return; `parse_recommendation`
    # ([ADR 0024]) accepts only this, so the prompt must ask for exactly it.
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    assert "oracle_id" in messages.system
    assert "role" in messages.system
    assert "rationale" in messages.system
    assert "JSON" in messages.system


def test_the_card_set_is_stated_as_a_closed_choice() -> None:
    # Closed vocabulary is the one rule the parser enforces by raising
    # ([ADR 0024]); the prompt states it so a valid reply is the easy path.
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    assert "only from the cards" in messages.system.lower()


def test_the_messages_match_the_complete_signature() -> None:
    # `LLMClient.complete(*, system, user)` — the return carries those two
    # strings under those two names, so the caller can hand them straight over.
    messages = build_messages(_REQUEST, _TEMPLATE, _CARDS)

    assert isinstance(messages.system, str) and messages.system
    assert isinstance(messages.user, str) and messages.user
    assert messages.system != messages.user


def test_the_prompt_scaffolding_carries_no_legality_colour_or_platform_language() -> None:
    # ADR 0001: legality, colour identity, and platform are deterministic
    # filters applied before any card reaches the model — the pool it is handed
    # already satisfies them, so there is nothing to check and only an exception
    # to hallucinate. The check is over what `build_messages` itself writes, so
    # the template is left empty and no cards are passed: a real template
    # legitimately *describes* its format (commander.md names "color identity")
    # and a real card's text may say anything, and both are handed through whole
    # ([ADR 0025]) rather than being instructions the prompt adds.
    messages = build_messages(_REQUEST, template="", cards=())
    scaffolding = (messages.system + messages.user).lower()

    for forbidden in ("legal", "banned", "color", "colour", "platform", "arena", "mtgo"):
        assert forbidden not in scaffolding


def test_the_prompt_scaffolding_offers_no_ranking_cue() -> None:
    # ADR 0008: a candidate's fused score and its channel distances are display
    # only. Neither is a judgement about theme fit, so neither is shown to the
    # model — and ADR 0005 wants a checkable argument rather than a number back,
    # so the prompt asks for no rating either. The template is left out because
    # commander.md's own framing legitimately uses the word "ranking".
    messages = build_messages(_REQUEST, template="", cards=_CARDS)
    prompt = (messages.system + messages.user).lower()

    for forbidden in ("score", "distance", "rank", "similarity"):
        assert forbidden not in prompt


def test_the_real_template_is_handed_through_untouched() -> None:
    # The whole template travels into the prompt verbatim ([ADR 0025]) — even
    # the Workflow section written for the planner and the parts describing the
    # format's colour-identity rule, which the prompt never turns into an
    # instruction. build_messages doesn't edit or filter it.
    template = (
        "# Commander\n\n"
        "100-card singleton decks built within one commander's color identity.\n\n"
        "## Heuristics\n- Ramp: ~10\n\n"
        "## Workflow\n1. Read the theme.\n"
    )

    messages = build_messages(_REQUEST, template, _CARDS)

    assert template in messages.system
