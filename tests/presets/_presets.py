"""Discovery helpers for the preset YAML files shipped under config/."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

CONFIG_DIR: Path = Path(__file__).resolve().parents[2] / "config"
PAPER_DIR: Path = CONFIG_DIR / "benchmark" / "paper_experiments"
SMOKE_DIR: Path = CONFIG_DIR / "benchmark" / "smoke"

SOURCE_PRESET_NAME = "source.yaml"

PaperFamily = Literal["piano", "guitar", "sweep"]

# Paper experiment presets are identified by their file-name prefix; the
# prefixes are disjoint, so the first match wins.
_PAPER_FAMILY_PREFIXES: tuple[tuple[str, PaperFamily], ...] = (
    ("piano_only", "piano"),
    ("guitar_only", "guitar"),
    ("transcriber_params_", "sweep"),
)

_TUNING_SMOKE_PREFIX = "tuning_"
_TUNING_SMOKE_SUFFIX = "_test.yaml"


def presets() -> list[Path]:
    """Every preset YAML under config/, excluding the reference source.yaml."""
    return sorted(
        path for path in CONFIG_DIR.rglob("*.yaml") if path.name != SOURCE_PRESET_NAME
    )


def paper_presets() -> list[Path]:
    """Every preset YAML directly in the paper experiments directory."""
    return sorted(PAPER_DIR.glob("*.yaml"))


def paper_family(path: Path) -> PaperFamily:
    """Classify a paper experiment preset by its file-name prefix."""
    for prefix, family in _PAPER_FAMILY_PREFIXES:
        if path.stem.startswith(prefix):
            return family
    raise ValueError(f"unrecognised paper experiment preset: {path}")


def tuning_smoke_pairs() -> list[tuple[Path, Path]]:
    """Pair each tuning smoke preset with its matching paper experiment preset."""
    pattern = f"{_TUNING_SMOKE_PREFIX}*{_TUNING_SMOKE_SUFFIX}"
    pairs: list[tuple[Path, Path]] = []
    for smoke in sorted(SMOKE_DIR.glob(pattern)):
        family = smoke.name[len(_TUNING_SMOKE_PREFIX) : -len(_TUNING_SMOKE_SUFFIX)]
        paper = PAPER_DIR / f"{family}_only.yaml"
        if not paper.is_file():
            raise ValueError(
                f"no paper experiment preset for tuning smoke preset: {smoke}"
            )
        pairs.append((smoke, paper))
    return sorted(pairs)