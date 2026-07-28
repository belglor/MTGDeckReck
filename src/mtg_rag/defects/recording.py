"""An `LLMClient` that writes every exchange to a file as it goes.

Wraps another client and returns its answers untouched, so nothing downstream
can tell the difference. That is the point of [ADR 0021]'s seam: a sweep can
record what the model was asked and what it said without `plan()`, `curate()`
or their parsers gaining a debug parameter, and without the validate-or-raise
contracts ([ADR 0022], [ADR 0024]) changing shape.

Recording the *response* is what makes this worth having. A failed run reports
only that the output never validated and why the parser rejected it; the text
that was rejected is discarded by the retry loop, so the one artifact needed to
tell "the model wrote prose around the JSON" from "it invented a card id" is
gone. Here it is on disk.

Records are self-identifying rather than labelled by the caller: the user
message opens with the stage's own preamble and then the theme, so a reader can
tell a plan call from a curation call, and which theme it belongs to, without
this wrapper being told either. Keeping it stateless means a sweep cannot forget
to update a label and mislabel an hour of output.

One JSON object per line, appended and flushed per call, so a run that is
interrupted still leaves a readable log of everything up to that point.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

from mtg_rag.llm import LLMClient


class RecordingClient:
    """Delegate to `inner`, appending each exchange to `path` as JSONL."""

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._calls = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def calls(self) -> int:
        """How many exchanges have been recorded."""
        return self._calls

    def complete(self, *, system: str, user: str) -> str:
        """Ask `inner`, record the exchange, and return its answer unchanged."""
        started = time.perf_counter()
        response = self._inner.complete(system=system, user=user)
        seconds = time.perf_counter() - started
        self._calls += 1
        self.write(
            {
                "record": "exchange",
                "call": self._calls,
                # Per-call timing, because decode dominates and nothing else
                # measures it: a curation call that runs long is the signal
                # that it is generating to its token cap rather than to EOS.
                "seconds": round(seconds, 1),
                "system": system,
                "user": user,
                "response": response,
                "response_chars": len(response),
            }
        )
        return response

    def write(self, record: Mapping[str, object]) -> None:
        """Append one record, flushed, so an interrupted run keeps what it had.

        Public because the sweep writes its own per-run records to the same
        file: a reader following one theme wants its exchanges and its scores
        interleaved, not in two files to be zipped together by hand.
        """
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
