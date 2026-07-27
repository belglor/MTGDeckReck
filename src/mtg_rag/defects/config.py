"""Defect-check configuration — values only.

The card-type vocabulary the rationale checks read. Lives here rather than in
the module that uses it so extending it is an edit to data ([CLAUDE.md]'s
config-module rule).
"""

from __future__ import annotations

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
