"""Tests for assembling the planner's prompt from a request and a template.

`build_messages` is the pure step [ADR 0023] leaves before the model call: the
whole format template plus the user's theme become the `system`/`user` strings
`LLMClient.complete` takes ([ADR 0021]). It asks the model for the typed
`[{query_text, purpose}]` plan ([ADR 0004]) and nothing else ([ADR 0022]), and
it never mentions legality, colour identity, or platform — those are
deterministic retrieval filters, not model reasoning ([ADR 0001]).

Nothing here touches a model or the filesystem: the template arrives as a
string, so these tests pass one directly.
"""

from __future__ import annotations

from mtg_rag.plan.prompt import build_messages

# A realistic theme and a stand-in template. The template text is a sentinel so
# a test can assert the function embeds *the argument it was handed*, not a file
# it read itself — the read lives in the planner function, a later issue.
_REQUEST = "a spooky graveyard deck that mills itself"
_TEMPLATE = "## Heuristics\n- Ramp: ~10\n- Card draw: ~10\n- Theme cards: the rest"


def test_the_template_text_is_embedded_verbatim() -> None:
    # Proves purity: the function consumes the passed string, so it can't be
    # reaching for a file of its own.
    messages = build_messages(_REQUEST, _TEMPLATE)

    assert _TEMPLATE in messages.system + messages.user


def test_the_request_text_is_embedded_verbatim() -> None:
    messages = build_messages(_REQUEST, _TEMPLATE)

    assert _REQUEST in messages.system + messages.user


def test_the_output_schema_is_specified() -> None:
    # The model is told the exact shape to return; the parser ([ADR 0022])
    # accepts only this, so the prompt must ask for exactly it.
    messages = build_messages(_REQUEST, _TEMPLATE)
    prompt = messages.system + messages.user

    assert "query_text" in prompt
    assert "purpose" in prompt
    assert "JSON" in prompt


def test_the_messages_match_the_complete_signature() -> None:
    # `LLMClient.complete(*, system, user)` — the return carries those two
    # strings under those two names, so the caller can hand them straight over.
    messages = build_messages(_REQUEST, _TEMPLATE)

    assert isinstance(messages.system, str) and messages.system
    assert isinstance(messages.user, str) and messages.user
    assert messages.system != messages.user


def test_the_prompt_scaffolding_carries_no_legality_colour_or_platform_language() -> None:
    # ADR 0001: legality, colour identity, and platform are deterministic
    # filters applied before any card reaches the model — there is no reasoning
    # for the model to add, only an exception for it to hallucinate. The check
    # is over what `build_messages` itself writes, so the template is left empty:
    # a real template legitimately *describes* its format (commander.md names
    # "colour identity"), and that descriptive text is handed through whole
    # ([ADR 0023]), not an instruction the prompt adds.
    messages = build_messages(_REQUEST, template="")
    scaffolding = (messages.system + messages.user).lower()

    for forbidden in ("legal", "banned", "color", "colour", "platform", "arena", "mtgo"):
        assert forbidden not in scaffolding


def test_the_real_template_is_handed_through_untouched() -> None:
    # The whole template travels into the prompt verbatim ([ADR 0023]) — even
    # the parts describing the format's colour-identity rule, which the prompt
    # never turns into an instruction. build_messages doesn't edit or filter it.
    template = (
        "# Commander\n\n"
        "100-card singleton decks built within one commander's colour identity.\n\n"
        "## Heuristics\n- Ramp: ~10\n"
    )

    messages = build_messages(_REQUEST, template)

    assert template in messages.system
