"""Assemble the planner's prompt from the format template and the user's theme.

A pure step — template text plus request text in, the `system`/`user` strings
`LLMClient.complete` takes out ([ADR 0021]). No file read (the planner reads the
template; here it is already a string) and no model, so the whole thing is a
string transform a test can pin down exactly.

What the prompt asks for, and what it deliberately leaves out:

- The whole template goes in verbatim ([ADR 0023]): the roles a deck covers plus
  the framing and workflow prose. The model decides which searches cover those
  roles and how many to run — that choice is the model's, not the app's
  ([ADR 0004]).
- The output contract is the typed `[{query_text, purpose}]` schema and nothing
  else: JSON only, no prose, no code fences, to give the one-shot parse
  ([ADR 0022]) the best chance on the first try.
- Legality, colour identity, and platform never appear. They are deterministic
  retrieval filters the user sets, applied before any card reaches the model
  ([ADR 0001]) — text about them here would only invite a hallucinated
  exception. The template may *describe* its format in those terms; that
  descriptive prose rides along untouched, but the prompt adds no instruction of
  its own.
"""

from __future__ import annotations

from dataclasses import dataclass

# The role and task. The model turns a theme into searches that cover the
# template's roles; it owns which searches and how many ([ADR 0004]).
_SYSTEM_INTRO = (
    "You are a deckbuilding planner for Magic: The Gathering. Turn the player's "
    "theme into a set of search queries that will retrieve candidate cards for "
    "their deck.\n\n"
    "Use the format guide below. It lists the roles a deck must cover and how to "
    "think about the format. Cover those roles for the player's theme, letting "
    "the theme absorb any role it naturally fills. You decide which searches to "
    "run and how many — a narrow theme needs only a few, a broad one more."
)

#: The `query_text` values the output contract's worked example shows below.
#: Named rather than left inline because #72 records the model copying them
#: back verbatim as its most frequent failure, and the check that counts that
#: ([ADR 0026]) has to match the strings the prompt actually shows.
#: `test_plan_prompt` pins these to the contract text, so editing the example
#: without updating them fails a test rather than silently blinding the check.
EXAMPLE_QUERIES = ("sacrifice for value", "mana rocks")

# The output contract. Real braces here, so this stays a plain string the
# function concatenates rather than a `.format` template needing them escaped.
_OUTPUT_CONTRACT = (
    "Return ONLY a JSON array. Each element is an object with exactly two "
    "string keys:\n"
    '- "query_text": what to search for\n'
    '- "purpose": the role this search covers, e.g. "ramp", "card draw", '
    '"theme payoff"\n\n'
    'Example: [{"query_text": "sacrifice for value", "purpose": "theme payoff"}, '
    '{"query_text": "mana rocks", "purpose": "ramp"}]\n\n'
    "Output the JSON array and nothing else: no prose, no explanation, no "
    "markdown code fences."
)

#: Introduces the user's theme in the user message. Kept apart from the theme
#: itself so the request text rides in unedited.
_USER_PREAMBLE = "Plan the searches for this deck theme:"


@dataclass(frozen=True, slots=True)
class PlannerMessages:
    """The two strings `LLMClient.complete(*, system, user)` takes.

    Named fields rather than a bare tuple so the caller hands them over by name
    and can't transpose them.
    """

    system: str
    user: str


def build_messages(request: str, template: str) -> PlannerMessages:
    """Build the planner's `system`/`user` messages from the request and template.

    Pure: `template` is the already-read template text and `request` is the
    user's free-text theme, so this is a string transform with no I/O and no
    model. Both arrive verbatim in the output ([ADR 0023]).
    """
    system = f"{_SYSTEM_INTRO}\n\nFormat guide:\n{template}\n\n{_OUTPUT_CONTRACT}"
    user = f"{_USER_PREAMBLE}\n\n{request}"
    return PlannerMessages(system=system, user=user)
