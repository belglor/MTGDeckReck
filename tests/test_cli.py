"""Tests for the shared entry-point helpers."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from mtg_rag.cli import use_utf8_stdout


def test_stdout_is_reconfigured_to_replace_undecodable_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # errors="replace" rather than the default: a mangled em-dash in a progress
    # line beats crashing part-way through a long run.
    captured: dict[str, Any] = {}

    class Reconfigurable:
        def reconfigure(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(sys, "stdout", Reconfigurable())
    use_utf8_stdout()

    assert captured == {"encoding": "utf-8", "errors": "replace"}


def test_is_a_noop_when_stdout_cannot_be_reconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # Captured stdout — as under pytest — has no reconfigure at all, and a CLI
    # must not fall over on its first line because of that.
    class Captured:
        pass

    monkeypatch.setattr(sys, "stdout", Captured())
    use_utf8_stdout()  # must not raise
