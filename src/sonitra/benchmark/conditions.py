from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sonitra.config import BenchmarkSection, PipelineConfig


@dataclass(frozen=True)
class Condition:
    """A named experimental condition: a set of dotted-path config overrides."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Filesystem-safe variant of the condition name."""
        return re.sub(r"[^A-Za-z0-9._=-]+", "_", self.name)


def expand_conditions(section: BenchmarkSection) -> list[Condition]:
    """Expand the benchmark section into a flat list of conditions.

    The baseline (no overrides) comes first, then explicit conditions, then
    one condition per sweep value. Sweeps express the parameter-influence
    axes: each value becomes a condition named '<axis>=<value>'.
    """
    conditions: list[Condition] = []
    if section.include_baseline:
        conditions.append(Condition(name=section.baseline_name))
    for condition in section.conditions:
        conditions.append(Condition(name=condition.name, overrides=dict(condition.overrides)))
    for sweep in section.sweeps:
        axis = sweep.name or sweep.parameter.rsplit(".", 1)[-1]
        for value in sweep.values:
            conditions.append(
                Condition(name=f"{axis}={value}", overrides={sweep.parameter: value})
            )
    names = [condition.name for condition in conditions]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate condition names: {sorted(duplicates)}")
    return conditions


def apply_overrides(config: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    """Return a new validated config with dotted-path overrides applied.

    Paths address nested sections and list elements by index, e.g.
    'pedalboard.effects.1.wet_level' or 'pipeline.sample_rate'.
    """
    data = config.model_dump(mode="python")
    for path, value in overrides.items():
        _set_path(data, path, value)
    return PipelineConfig.model_validate(data)


def _set_path(data: Any, path: str, value: Any) -> None:
    segments = path.split(".")
    target = data
    for position, segment in enumerate(segments[:-1]):
        target = _descend(target, segment, path)
        if target is None:
            raise KeyError(f"Override path '{path}' hit a null value at '{segment}'")
    leaf = segments[-1]
    if isinstance(target, list):
        target[_list_index(target, leaf, path)] = value
    elif isinstance(target, dict):
        if leaf not in target:
            raise KeyError(f"Override path '{path}' has unknown key '{leaf}'")
        target[leaf] = value
    else:
        raise KeyError(f"Override path '{path}' cannot descend into {type(target).__name__}")


def _descend(target: Any, segment: str, path: str) -> Any:
    if isinstance(target, list):
        return target[_list_index(target, segment, path)]
    if isinstance(target, dict):
        if segment not in target:
            raise KeyError(f"Override path '{path}' has unknown key '{segment}'")
        return target[segment]
    raise KeyError(f"Override path '{path}' cannot descend into {type(target).__name__}")


def _list_index(target: list, segment: str, path: str) -> int:
    try:
        index = int(segment)
    except ValueError as exc:
        raise KeyError(f"Override path '{path}' needs a list index, got '{segment}'") from exc
    if not -len(target) <= index < len(target):
        raise KeyError(f"Override path '{path}' index {index} out of range")
    return index
