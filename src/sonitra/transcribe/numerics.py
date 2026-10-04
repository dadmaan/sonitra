"""Numeric settings shared by the neural transcription backends, passed through the process environment."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

ENV_NUMERIC_MODE = "SONITRA_NUMERIC_MODE"
ENV_GPU_MEMORY_GROWTH = "SONITRA_GPU_MEMORY_GROWTH"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _parse_growth(raw: str | None) -> bool:
    """Whether a raw environment value asks for GPU memory growth."""
    return raw is not None and raw.strip().lower() in _TRUTHY


def read_numeric_env() -> tuple[str, bool]:
    """Process default (numeric_mode, gpu_memory_growth) every builder reads.

    The mode is returned verbatim: each backend lowercases and validates it in
    its own constructor. Calling ``make_transcriber`` directly, outside a
    benchmark or ``transcribe`` run, reads these variables directly and so gets
    ``off`` / ``False`` whenever they are unset.
    """
    return (
        os.environ.get(ENV_NUMERIC_MODE, "off"),
        _parse_growth(os.environ.get(ENV_GPU_MEMORY_GROWTH)),
    )


def _warn_on_difference(
    var: str, raw: str, key: str, config_value: str | bool
) -> None:
    """Report an already-exported value that the config overrides."""
    logger.warning(
        "%s=%r in the environment differs from transcription.%s=%r in the config; "
        "using the config value",
        var,
        raw,
        key,
        config_value,
    )


@contextmanager
def numeric_env(numeric_mode: str, gpu_memory_growth: bool) -> Iterator[None]:
    """Publish the config's numeric settings to backend builders for this block.

    The environment is only the transport from the config to the builders, so a
    value already exported from elsewhere is reported and replaced, and the
    caller's environment is left exactly as it was found.
    """
    settings = {
        ENV_NUMERIC_MODE: numeric_mode,
        ENV_GPU_MEMORY_GROWTH: "1" if gpu_memory_growth else "0",
    }
    # Growth is compared after parsing, because "true" reaches a backend as
    # True and "0" as False; the mode is compared as read, unnormalised.
    compared = {
        ENV_NUMERIC_MODE: ("numeric_mode", numeric_mode, lambda raw: raw),
        ENV_GPU_MEMORY_GROWTH: (
            "gpu_memory_growth",
            gpu_memory_growth,
            _parse_growth,
        ),
    }
    previous = {var: os.environ.get(var) for var in settings}

    for var, (key, config_value, parse) in compared.items():
        raw = previous[var]
        if raw is not None and parse(raw) != config_value:
            _warn_on_difference(var, raw, key, config_value)

    os.environ.update(settings)
    try:
        yield
    finally:
        for var, raw in previous.items():
            if raw is None:
                del os.environ[var]
            else:
                os.environ[var] = raw