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

# ── per-backend guards: strict is fatal, a fallback is reported once ─────


def _fake_gpu(tf: object, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Report one GPU to the guard and return the list it records growth calls in."""
    monkeypatch.setattr(
        tf.config, "list_physical_devices", lambda kind: [object()], raising=False
    )
    monkeypatch.setattr(tf.config.experimental, "enable_op_determinism", lambda: None)
    monkeypatch.setattr(
        tf.config.experimental, "enable_tensor_float_32_execution", lambda flag: None
    )
    return []


def test_basic_pitch_strict_failure_raises_every_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf = pytest.importorskip("tensorflow")

    from sonitra.transcribe import basic_pitch
    from sonitra.transcribe.base import NumericSettingsError

    _fake_gpu(tf, monkeypatch)

    def set_memory_growth(gpu: object, grow: bool) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(tf.config.experimental, "set_memory_growth", set_memory_growth)

    # The state stays unset on a strict failure, so the next file must fail the
    # same way instead of inheriting an unverified "already applied" answer.
    for _ in range(2):
        with pytest.raises(NumericSettingsError):
            basic_pitch._apply_numeric_settings("strict", True)


def test_basic_pitch_warn_failure_returns_fallbacks_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tf = pytest.importorskip("tensorflow")

    from sonitra.transcribe import basic_pitch

    calls = _fake_gpu(tf, monkeypatch)

    def set_memory_growth(gpu: object, grow: bool) -> None:
        calls.append(gpu)
        raise RuntimeError("boom")

    monkeypatch.setattr(tf.config.experimental, "set_memory_growth", set_memory_growth)

    with caplog.at_level("WARNING", logger="sonitra.transcribe.basic_pitch"):
        first = basic_pitch._apply_numeric_settings("warn", True)
        second = basic_pitch._apply_numeric_settings("warn", True)

    assert len(first) == 1
    assert first[0].startswith("memory growth:")
    assert second == first
    assert len(calls) == 1, "the framework was called again for settings already resolved"
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "sonitra.transcribe.basic_pitch"
    ]
    assert len(warnings) == 1


def test_basic_pitch_success_returns_no_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf = pytest.importorskip("tensorflow")

    from sonitra.transcribe import basic_pitch

    _fake_gpu(tf, monkeypatch)
    monkeypatch.setattr(
        tf.config.experimental, "set_memory_growth", lambda gpu, grow: None
    )

    assert basic_pitch._apply_numeric_settings("warn", True) == ()


def test_basic_pitch_default_settings_skip_tensorflow_import() -> None:
    import subprocess
    import sys

    code = (
        "from sonitra.transcribe import basic_pitch; "
        "import sys; "
        "basic_pitch._apply_numeric_settings('off', False); "
        "assert 'tensorflow' not in sys.modules, 'tensorflow imported for off/False'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_transkun_strict_failure_raises_every_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    from sonitra.transcribe import torch_support
    from sonitra.transcribe.base import NumericSettingsError

    def boom(mode: bool, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", boom)

    for _ in range(2):
        with pytest.raises(NumericSettingsError):
            torch_support.apply_torch_numeric_settings(
                "strict", False, backend="transkun"
            )


def test_transkun_warn_only_typeerror_is_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    from sonitra.transcribe import torch_support

    real = torch.use_deterministic_algorithms

    def without_warn_only(mode: bool, **kwargs: object) -> None:
        if "warn_only" in kwargs:
            raise TypeError("unexpected keyword argument 'warn_only'")
        real(mode, **kwargs)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", without_warn_only)

    fallbacks = torch_support.apply_torch_numeric_settings(
        "warn", False, backend="transkun"
    )

    assert len(fallbacks) == 1
    assert "warn_only" in fallbacks[0]
