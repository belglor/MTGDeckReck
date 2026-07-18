"""Resolve the hard constraints a retrieval pass runs under.

A deck request arrives as free text ("spooky graveyard shenanigans in Golgari,
Commander"). Before we can retrieve anything we need two things pinned down:
which format the deck is for, and which colors it may use. Asking the user to
fill in two more pickers before they can type a theme is friction, so we infer
both from the query itself and let the retrieval pass filter on the result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# The five color-identity letters Scryfall uses, in WUBRG order.
WUBRG = ("W", "U", "B", "R", "G")

SUPPORTED_FORMATS = ("commander", "modern", "pioneer", "legacy", "vintage")

_INFERENCE_PROMPT = """\
Read the deck request below and decide two things:

1. `format` — which Magic format the deck is for. One of: {formats}.
   If the request does not say, answer "commander".
2. `colors` — the deck's color identity, as a list of WUBRG letters.
   Infer it from any guild or shard names ("Golgari" -> ["B", "G"]),
   explicit color words, or the flavour of the theme itself. An empty
   list means colorless.

Answer with JSON only: {{"format": "...", "colors": ["..."]}}

Deck request:
{query}
"""


class CompletionClient(Protocol):
    """Minimal surface we need from whatever LLM client is wired in."""

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Constraints:
    """The filters a retrieval pass applies before scoring anything."""

    deck_format: str
    colors: tuple[str, ...]

    def permits(self, card_colors: Sequence[str]) -> bool:
        """True when every color on the card is inside the deck's identity."""
        return all(color in self.colors for color in card_colors)


def infer_constraints(query: str, client: CompletionClient) -> Constraints:
    """Ask the model to pull format and color identity out of the raw query."""
    prompt = _INFERENCE_PROMPT.format(
        formats=", ".join(SUPPORTED_FORMATS),
        query=query.strip(),
    )
    answer: dict[str, object] = json.loads(client.complete(prompt))

    raw_format = answer.get("format", "commander")
    deck_format = raw_format.lower() if isinstance(raw_format, str) else "commander"

    raw_colors = answer.get("colors", [])
    colors: list[str] = []
    if isinstance(raw_colors, list):
        for entry in raw_colors:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(entry, str) and entry.upper() in WUBRG:
                colors.append(entry.upper())

    return Constraints(deck_format=deck_format, colors=tuple(colors))
