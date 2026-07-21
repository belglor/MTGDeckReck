"""Vector store configuration."""

from __future__ import annotations

#: Collection names are this prefix plus the channel — "cards_oracle". One
#: collection per channel, because the channels index different card counts
#: ([ADR 0014]) and are searched independently before fusion ([ADR 0008]).
COLLECTION_PREFIX = "cards_"

#: Distance space. Chroma's default is `l2`; the encoder emits unit-norm
#: vectors and ranking is cosine, so this is set explicitly on every collection.
DISTANCE_SPACE = "cosine"

#: Chroma's product telemetry posts to a third party by default. Off, so
#: nothing about this corpus or the queries run against it leaves the machine.
ANONYMIZED_TELEMETRY = False
