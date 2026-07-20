"""Tests for ingest configuration and its `.env` loader.

Nothing here touches the network or the real `.env`/environment beyond what
each test sets up and `monkeypatch` tears back down.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from mtg_rag.ingest import config as config_module


@pytest.fixture(autouse=True)
def _reset_config_module() -> Iterator[None]:
    """Reload after every test so an env override doesn't leak into the next."""
    yield
    importlib.reload(config_module)


# --- USER_AGENT --------------------------------------------------------------


def test_default_user_agent_carries_no_contact_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRYFALL_USER_AGENT", raising=False)
    importlib.reload(config_module)
    assert "@" not in config_module.USER_AGENT
    assert "github.com" not in config_module.USER_AGENT


def test_env_var_overrides_the_default_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYFALL_USER_AGENT", "MTGDeckReck/0.1 (test@example.com)")
    importlib.reload(config_module)
    assert config_module.USER_AGENT == "MTGDeckReck/0.1 (test@example.com)"


# --- load_dotenv --------------------------------------------------------------


def test_load_dotenv_populates_a_missing_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MTGDECKRECK_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MTGDECKRECK_TEST_VAR=from-dotenv\n", encoding="utf-8")

    config_module.load_dotenv(env_file)

    assert os.environ["MTGDECKRECK_TEST_VAR"] == "from-dotenv"


def test_load_dotenv_does_not_override_a_real_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MTGDECKRECK_TEST_VAR", "from-environment")
    env_file = tmp_path / ".env"
    env_file.write_text("MTGDECKRECK_TEST_VAR=from-dotenv\n", encoding="utf-8")

    config_module.load_dotenv(env_file)

    assert os.environ["MTGDECKRECK_TEST_VAR"] == "from-environment"


def test_load_dotenv_skips_comments_and_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MTGDECKRECK_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\n\nMTGDECKRECK_TEST_VAR=value\n", encoding="utf-8")

    config_module.load_dotenv(env_file)

    assert os.environ["MTGDECKRECK_TEST_VAR"] == "value"


def test_load_dotenv_is_a_noop_for_a_missing_file(tmp_path: Path) -> None:
    config_module.load_dotenv(tmp_path / "nope.env")  # must not raise
