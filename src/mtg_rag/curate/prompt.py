"""Assemble curation's prompt from the format template, the theme, and the pool.

A pure step — template text, request text, and the retrieved candidates in, the
`system`/`user` strings `LLMClient.complete` takes out ([ADR 0021]). No file read
(the curation entry point reads the template), no corpus read (the pool arrives
hydrated), and no model, so the whole thing is a string transform a test can pin
down exactly.

What the prompt asks for, and what it deliberately leaves out:

- The whole template goes in verbatim ([ADR 0025]), the Workflow section written
  for the planner included. Curation reads its framing and its composition
  targets out of it.
- The candidates are a **closed set**: the model chooses among them and copies
  each `oracle_id` back, because the parser checks every returned id against the
  pool and raises on one that isn't a member ([ADR 0024]). Omitting a candidate
  is fine — selecting down is the job.
- Each candidate carries the `purpose` hypotheses of the searches that found it.
  That is curation's starting guess at the card's role ([ADR 0005]), not an
  answer: a card retrieved as "ramp" may well belong under the theme.
- The output contract is the typed `[{oracle_id, role, rationale}]` schema and
  nothing else: JSON only, no prose, no code fences, to give the one-shot parse
  ([ADR 0024]) the best chance on the first try.
- Copying the card's own rules or flavor text into a rationale is forbidden
  outright ([#106]). It argues nothing the player cannot already read, and it is
  what made replies long enough to hit the generation cap and be cut mid-string.
- The contract states a reply budget, derived from `MAX_NEW_TOKENS` so the two
  cannot drift. The cap is otherwise silent: overrunning it truncates the JSON,
  and the parser's complaint is indistinguishable from a model that cannot
  follow the schema.
- No number reaches the model. A candidate's fused score and its channel
  distances are display-only ([ADR 0008]) and say nothing about theme fit, so
  they are not shown; and the answer asked for is a role plus a checkable
  argument, never a rating ([ADR 0005]).
- Legality, colour identity, and platform never appear. They are deterministic
  retrieval filters the user sets, already satisfied by every card in the pool
  ([ADR 0001]) — text about them here would only invite a hallucinated
  exception. The template may *describe* its format in those terms, and a card's
  own text may say anything; both ride along untouched, but the prompt adds no
  instruction of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mtg_rag.curate.config import RATIONALE_BUDGET_FRACTION
from mtg_rag.llm_config import MAX_NEW_TOKENS

# The role and task. The model selects from a fixed pool, assigns each pick a
# role, and argues its theme fit ([ADR 0005]).
_SYSTEM_INTRO = (
    "You are a deckbuilding curator for Magic: The Gathering. You are given a "
    "player's theme and the candidate cards retrieved for it. Choose the cards "
    "that belong in the deck, say what job each one does, and argue why it fits "
    "the theme.\n\n"
    "Use the format guide below. It says what a deck of this format needs and "
    "how to think about the format. Build for theme and flavour: prefer the card "
    "that says something about the player's idea over the merely powerful one."
)

#: The `rationale` values the output contract's worked example shows below.
#: Named rather than left inline because #91 found this exact sentence returned
#: as the top card's rationale in 5 of 5 runs — including three where mill had
#: nothing to do with the theme — and the check that counts that ([ADR 0026])
#: has to match the string the prompt actually shows. `test_curate_prompt` pins
#: these to the contract text, so rewording the example without updating them
#: fails a test rather than silently blinding the check.
EXAMPLE_RATIONALES = ("Mills you every upkeep, filling the graveyard the deck feeds on.",)

#: The reply budget the contract states, derived so it tracks the real cap
#: rather than drifting from it ([#106]). Stated to the model because the cap
#: itself is silent: generation stops mid-string and the parser reports a
#: malformed reply, which looks identical to a model that cannot follow the
#: schema.
_REPLY_TOKEN_BUDGET = int(MAX_NEW_TOKENS * RATIONALE_BUDGET_FRACTION)

# The output contract. Real braces here, so this stays a plain string the
# function concatenates rather than a `.format` template needing them escaped.
_OUTPUT_CONTRACT = (
    "Choose only from the cards listed in the next message. Every card you "
    "return must be one of them, with its oracle_id copied exactly as given. "
    "Leaving a candidate out is fine — not every candidate belongs in the "
    "deck.\n\n"
    "Return ONLY a JSON array. Each element is an object with exactly three "
    "string keys:\n"
    '- "oracle_id": the chosen card\'s id, copied from the list\n'
    '- "role": the job it does in the deck, e.g. "ramp", "card draw", '
    '"theme payoff"\n'
    '- "rationale": one sentence on why this card fits the theme, naming what '
    "the card actually does so the player can check the claim\n\n"
    'Example: [{"oracle_id": "1a2b3c", "role": "theme payoff", "rationale": '
    '"Mills you every upkeep, filling the graveyard the deck feeds on."}]\n\n'
    "Give the same card at most one entry, and add no other key — no rating, no "
    "number. The role and the argument are the whole answer.\n\n"
    "Never copy the card's own text into the rationale. Not its rules text, not "
    "its flavor text, not a phrase from either. The player can already read the "
    "card; quoting it back argues nothing. Say in your own words what the card "
    "does and why that serves this theme.\n\n"
    "Keep every rationale to one short sentence. Your whole reply must fit in "
    + str(_REPLY_TOKEN_BUDGET)
    + " tokens — a longer reply is cut off mid-word and cannot be read at all, "
    "so a shorter list of well-argued cards beats a long one that never "
    "arrives.\n\n"
    "Output the JSON array and nothing else: no prose, no explanation, no "
    "markdown code fences."
)

#: Introduces the user's theme in the user message. Kept apart from the theme
#: itself so the request text rides in unedited.
_USER_PREAMBLE = "Curate a deck for this theme:"

#: Introduces the closed set of candidates, and says what the trailing "found
#: for" line on each of them means.
_CARDS_PREAMBLE = (
    "The candidate cards. Each ends with the roles the searches that found it "
    "were looking for — a starting guess at its job here, not the answer:"
)


@dataclass(frozen=True, slots=True)
class CurationCard:
    """One candidate as curation sees it: the card's own text, and why it surfaced.

    The hydrated corpus fields plus the distinct `purpose`s gathered from the
    candidate's retrieval sources. Assembling this from a `Candidate` and its
    hydrated row is the caller's job, which keeps this module a string transform.

    `mana_cost`, `oracle_text`, and `flavor_text` are legitimately empty — a land
    has no cost, a vanilla creature no text — and empty means the line is left
    out. Notably absent: the fused score and the per-source distances, which are
    display-only and never a theme-fit signal ([ADR 0008]).
    """

    oracle_id: str
    name: str
    mana_cost: str
    type_line: str
    oracle_text: str
    flavor_text: str
    purposes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CurationMessages:
    """The two strings `LLMClient.complete(*, system, user)` takes.

    Named fields rather than a bare tuple so the caller hands them over by name
    and can't transpose them.
    """

    system: str
    user: str


def build_messages(request: str, template: str, cards: Sequence[CurationCard]) -> CurationMessages:
    """Build curation's `system`/`user` messages from the request, template, and pool.

    Pure: `template` is the already-read template text, `request` is the user's
    free-text theme, and `cards` is the already-hydrated pool, so this is a
    string transform with no I/O and no model. Template and request arrive
    verbatim in the output ([ADR 0025]); the cards keep the order they are given,
    which is the pool's order.
    """
    system = f"{_SYSTEM_INTRO}\n\nFormat guide:\n{template}\n\n{_OUTPUT_CONTRACT}"
    card_text = "\n\n".join(_render_card(card) for card in cards)
    user = f"{_USER_PREAMBLE}\n\n{request}\n\n{_CARDS_PREAMBLE}\n\n{card_text}"
    return CurationMessages(system=system, user=user)


def _render_card(card: CurationCard) -> str:
    """One candidate as a labelled block, empty fields omitted."""
    lines = [f"oracle_id: {card.oracle_id}", f"name: {card.name}"]
    if card.mana_cost:
        lines.append(f"mana cost: {card.mana_cost}")
    lines.append(f"type: {card.type_line}")
    if card.oracle_text:
        lines.append(f"text: {card.oracle_text}")
    if card.flavor_text:
        lines.append(f"flavor: {card.flavor_text}")
    lines.append(f"found for: {', '.join(card.purposes)}")
    return "\n".join(lines)
