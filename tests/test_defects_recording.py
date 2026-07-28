"""Tests for the recording `LLMClient` wrapper ([ADR 0026]).

The wrapper exists because a failed run reports only *that* the output never
validated: the retry loops in `plan()` and `curate()` discard the text the
parser rejected, so the one artifact needed to tell "the model wrapped prose
around its JSON" from "it invented a card id" is gone. Recording it costs
nothing downstream because `LLMClient` is a Protocol ([ADR 0021]).
"""

from __future__ import annotations

import json
from pathlib import Path


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.seen: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.seen.append((system, user))
        return self._responses.pop(0)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_the_wrapped_answer_is_returned_unchanged(tmp_path: Path) -> None:
    # Nothing downstream may be able to tell it is wrapped, or the sweep would
    # measure a different pipeline than the one that ships.
    from mtg_rag.defects.recording import RecordingClient

    client = RecordingClient(_FakeClient(["the answer"]), tmp_path / "log.jsonl")

    assert client.complete(system="s", user="u") == "the answer"


def test_the_prompt_and_reply_are_both_recorded(tmp_path: Path) -> None:
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    RecordingClient(_FakeClient(["a reply"]), path).complete(system="a system", user="a user")

    records = _records(path)
    assert len(records) == 1
    assert records[0]["record"] == "exchange"
    assert records[0]["call"] == 1
    assert records[0]["system"] == "a system"
    assert records[0]["user"] == "a user"
    assert records[0]["response"] == "a reply"
    assert records[0]["response_chars"] == len("a reply")


def test_each_exchange_carries_its_own_duration(tmp_path: Path) -> None:
    # The number that would have answered "why is one theme taking 16 minutes"
    # without tokenizing anything by hand: a call running long is the signal it
    # is generating to its token cap rather than stopping at EOS.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    RecordingClient(_FakeClient(["a reply"]), path).complete(system="s", user="u")

    assert isinstance(_records(path)[0]["seconds"], int | float)


def test_a_reply_that_would_fail_to_validate_is_still_recorded(tmp_path: Path) -> None:
    # The whole point: the retry loop throws this text away, so if it is not
    # captured here it is captured nowhere.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    client = RecordingClient(_FakeClient(["here you go: not json at all"]), path)
    client.complete(system="s", user="u")

    assert _records(path)[0]["response"] == "here you go: not json at all"


def test_every_retry_gets_its_own_record(tmp_path: Path) -> None:
    # A stage that retries once makes two calls ([ADR 0022], [ADR 0024]); both
    # belong in the log, or the second attempt looks like the only one.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    client = RecordingClient(_FakeClient(["bad", "also bad"]), path)
    client.complete(system="s", user="u")
    client.complete(system="s", user="u")

    records = _records(path)
    assert [r["call"] for r in records] == [1, 2]
    assert [r["response"] for r in records] == ["bad", "also bad"]
    assert client.calls == 2


def test_the_log_is_readable_after_each_call(tmp_path: Path) -> None:
    # Appended and flushed per call, so an interrupted run — an hour into a
    # sweep — still leaves everything up to that point.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    client = RecordingClient(_FakeClient(["one", "two"]), path)

    client.complete(system="s", user="u")
    assert len(_records(path)) == 1

    client.complete(system="s", user="u")
    assert len(_records(path)) == 2


def test_a_missing_directory_is_created(tmp_path: Path) -> None:
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "nested" / "deeper" / "log.jsonl"
    RecordingClient(_FakeClient(["x"]), path).complete(system="s", user="u")

    assert path.exists()


def test_card_text_survives_the_round_trip(tmp_path: Path) -> None:
    # Card names and flavor text carry non-ASCII (em dashes, accented names);
    # a log that mangles them is useless for reading a rejected reply.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    reply = "Séance — an apocalypse in dragon form… {1}{B}"
    RecordingClient(_FakeClient([reply]), path).complete(system="s", user="u")

    assert _records(path)[0]["response"] == reply


def test_a_run_record_can_share_the_log_with_the_exchanges(tmp_path: Path) -> None:
    # The sweep writes its own per-run records here, interleaved, so a reader
    # following one theme gets its exchanges and its scores in one file rather
    # than zipping two together by hand.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    client = RecordingClient(_FakeClient(["a reply"]), path)
    client.complete(system="s", user="u")
    client.write({"record": "run", "theme": "graveyard mill", "seconds": 12.3})

    records = _records(path)
    assert [r["record"] for r in records] == ["exchange", "run"]
    assert records[1]["theme"] == "graveyard mill"


def test_a_record_holding_an_unserializable_value_still_writes(tmp_path: Path) -> None:
    # `asdict` on a RunScores can carry a Path or similar; a debug log that
    # raises while recording a failure would destroy the artifact it exists for.
    from mtg_rag.defects.recording import RecordingClient

    path = tmp_path / "log.jsonl"
    client = RecordingClient(_FakeClient([]), path)
    client.write({"record": "run", "where": Path("data/cards.parquet")})

    assert "cards.parquet" in str(_records(path)[0]["where"])
