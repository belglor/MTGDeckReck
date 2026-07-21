"""Helpers shared by the `__main__` entry points."""

from __future__ import annotations

import io
import sys
from typing import Any, cast


def use_utf8_stdout() -> None:
    """Make stdout tolerate the corpus.

    Card names and flavor text are full of em-dashes and accents, and a cp1252
    console raises on the first one printed. Errors are replaced rather than
    raised: a mangled character in a progress line is a far better outcome than
    a crash part-way through a long run.

    A no-op where stdout has no `reconfigure` — which is the case whenever it
    has been captured, as under pytest.
    """
    stdout = cast("io.TextIOWrapper[Any]", sys.stdout)
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8", errors="replace")
