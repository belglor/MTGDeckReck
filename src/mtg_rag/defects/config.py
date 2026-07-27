"""Defect-check configuration — values only.

The card-type vocabulary the rationale checks read, and the theme set the sweep
runs. Both live here rather than in the modules that use them, so changing what
is measured is an edit to data ([CLAUDE.md]'s config-module rule).

The theme set is committed rather than passed in, for the reason `evals/config`
gives about the golden set: it is the definition of what the instrument
measures, and a sweep run against an ad-hoc list is not comparable to any other.
"""

from __future__ import annotations

from typing import Literal

#: Magic's card types, as they appear in a `type_line` and as a rationale would
#: name them. Supertypes (Legendary, Basic, Snow) and subtypes (Elf, Aura,
#: Equipment) are deliberately absent: a rationale calling a card "a powerful
#: Equipment" when it is an Artifact — Equipment is *correct*, and only the
#: eight types below are mutually exclusive enough that naming the wrong one is
#: a mistake rather than a shorthand.
CARD_TYPES = (
    "creature",
    "instant",
    "sorcery",
    "artifact",
    "enchantment",
    "land",
    "planeswalker",
    "battle",
)

#: How a theme is phrased. #72's sweep found the split load-bearing: evocative
#: requests degrade worse than rules-oriented ones, because a mechanical request
#: hands the model the Magic word it needs and an evocative one does not. Kept
#: on each theme so a later run can check whether that gap has closed.
type ThemeKind = Literal["thematic", "mechanical"]

#: The sweep's fixed theme set: #72's ten prompts, verbatim.
#:
#: Taken from #72 rather than #91 because they are the only prompts in this
#: project's history recorded well enough to re-run. #91's sweep names four of
#: its five themes only as shorthand (`dragon`, `angels`, `counters`,
#: `aristocrats`), so its exact wording is unrecoverable and its numbers cannot
#: be continued by anyone — the gap this constant exists to close. Curation runs
#: on whatever the planner returns, so one set serves both stages.
#:
#: Changing this list starts a new baseline. Numbers are comparable only within
#: a fixed theme set, the same rule [ADR 0011] applies to embedding config.
SWEEP_THEMES: tuple[tuple[str, ThemeKind], ...] = (
    ("a spooky graveyard deck that mills itself", "thematic"),
    ("a dragon tribal deck full of fire and flying", "thematic"),
    ("an elfball deck that goes wide with tokens", "thematic"),
    ("a pirates and treasure deck with a swashbuckling feel", "thematic"),
    ("a lifegain angels deck that feels holy and radiant", "thematic"),
    (
        "a deck that plays a lot with ETB effects and bouncing creatures back to hand",
        "mechanical",
    ),
    ("a deck built around +1/+1 counters and proliferate", "mechanical"),
    ("an artifact deck that combos with cheap artifacts and cost reduction", "mechanical"),
    ("a deck that wins by drawing extra cards and taking extra turns", "mechanical"),
    ("an aristocrats deck that drains life whenever my creatures die", "mechanical"),
)
