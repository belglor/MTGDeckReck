"""Embedding configuration.

Constants only. The channel vocabulary lives here rather than beside the
composition logic so `channels.py` holds behavior and nothing else, and so the
store and the CLI can name a channel without importing the composition module.

The three channels are settled in [ADR 0007]: oracle text (with the card name
folded in), flavor text, and type line. They are embedded separately because
each is a different register of language — terse rules prose, evocative
narrative, and a controlled vocabulary — and averaging them into one vector
lets the longest dominate and blurs the signal that makes each useful.
"""

from __future__ import annotations

from typing import Literal

#: One semantic channel. Doubles as the per-channel collection suffix.
type Channel = Literal["oracle", "flavor", "type"]

#: Every channel, in the order they are built and reported.
CHANNELS: tuple[Channel, ...] = ("oracle", "flavor", "type")

#: Name of the composed-text column in the frame `channel_frame` returns. The
#: encoder and the CLI read it, so it is named once here rather than spelled
#: out at each call site.
TEXT_COLUMN = "text"
