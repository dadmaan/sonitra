"""Numeric settings read from and published to the process environment."""
from __future__ import annotations

import os

import pytest

from sonitra.transcribe.numerics import (
    ENV_GPU_MEMORY_GROWTH,
    ENV_NUMERIC_MODE,
    numeric_env,
    read_numeric_env,
)

_LOGGER = "sonitra.transcribe.numerics"


def _numeric_warnings(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    """Messages this module logged that mention *needle*."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER and needle in record.getMessage()
    ]


def test_read_numeric_env_defaults_when_unset() -> None:
    assert read_numeric_env() == ("off", False)


def test_read_numeric_env_returns_mode_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NUMERIC_MODE, "STRICT")
    assert read_numeric_env()[0] == "STRICT"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_read_numeric_env_parses_growth_truthy_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, value)
    assert read_numeric_env()[1] is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  "])
def test_read_numeric_env_parses_growth_falsy_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, value)
    assert read_numeric_env()[1] is False


def test_numeric_env_sets_config_values_inside_block() -> None:
    with numeric_env("strict", True):
        assert read_numeric_env() == ("strict", True)


def test_numeric_env_restores_absent_variables() -> None:
    assert ENV_NUMERIC_MODE not in os.environ
    assert ENV_GPU_MEMORY_GROWTH not in os.environ

    with numeric_env("strict", True):
        assert os.environ[ENV_NUMERIC_MODE] == "strict"
        assert os.environ[ENV_GPU_MEMORY_GROWTH] == "1"

    assert ENV_NUMERIC_MODE not in os.environ
    assert ENV_GPU_MEMORY_GROWTH not in os.environ


def test_numeric_env_restores_previous_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NUMERIC_MODE, "off")
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, "0")

    with numeric_env("strict", True):
        assert read_numeric_env() == ("strict", True)

    assert os.environ[ENV_NUMERIC_MODE] == "off"
    assert os.environ[ENV_GPU_MEMORY_GROWTH] == "0"


def test_numeric_env_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_NUMERIC_MODE, "warn")
    entered = False

    # NotImplementedError is a RuntimeError subclass, so the guard also proves
    # the block body itself is what raised.
    with pytest.raises(RuntimeError):
        with numeric_env("strict", True):
            entered = True
            raise RuntimeError("boom")

    assert entered
    assert os.environ[ENV_NUMERIC_MODE] == "warn"


def test_numeric_env_config_wins_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENV_NUMERIC_MODE, "off")

    with caplog.at_level("WARNING", logger=_LOGGER):
        with numeric_env("strict", False):
            assert read_numeric_env()[0] == "strict"

    messages = _numeric_warnings(caplog, ENV_NUMERIC_MODE)
    assert len(messages) == 1
    message = messages[0]
    assert ENV_NUMERIC_MODE in message
    assert "'off'" in message
    assert "transcription.numeric_mode" in message
    assert "'strict'" in message
    assert "using the config value" in message


def test_numeric_env_growth_compared_parsed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, "true")

    with caplog.at_level("WARNING", logger=_LOGGER):
        with numeric_env("off", True):
            pass
    assert _numeric_warnings(caplog, ENV_GPU_MEMORY_GROWTH) == []

    caplog.clear()
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, "0")

    with caplog.at_level("WARNING", logger=_LOGGER):
        with numeric_env("off", True):
            pass
    messages = _numeric_warnings(caplog, ENV_GPU_MEMORY_GROWTH)
    assert len(messages) == 1
    assert "'0'" in messages[0]
    assert "True" in messages[0]


def test_numeric_env_matching_env_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENV_NUMERIC_MODE, "strict")
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, "1")

    with caplog.at_level("WARNING", logger=_LOGGER):
        with numeric_env("strict", True):
            pass

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER
    ] == []


def test_numeric_env_warns_on_each_entry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENV_NUMERIC_MODE, "off")
    monkeypatch.setenv(ENV_GPU_MEMORY_GROWTH, "0")

    with caplog.at_level("WARNING", logger=_LOGGER):
        with numeric_env("strict", True):
            pass
        with numeric_env("strict", True):
            pass

    messages = _numeric_warnings(caplog, ENV_NUMERIC_MODE) + _numeric_warnings(
        caplog, ENV_GPU_MEMORY_GROWTH
    )
    assert len(messages) == 4
    assert all(
        ENV_NUMERIC_MODE in message or ENV_GPU_MEMORY_GROWTH in message
        for message in messages
    )