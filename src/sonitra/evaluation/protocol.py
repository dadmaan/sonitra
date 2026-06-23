from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from sonitra.evaluation.types import NoteEvent

if TYPE_CHECKING:
    from sonitra.config import EvaluationSection


@runtime_checkable
class SymbolicMetric(Protocol):
    """A metric comparing reference and estimated note sequences."""

    name: str

    def compute(
        self, reference: Sequence[NoteEvent], estimate: Sequence[NoteEvent]
    ) -> dict[str, float]: ...


@runtime_checkable
class AudioMetric(Protocol):
    """A metric comparing reference and estimated audio signals."""

    name: str

    def compute(
        self,
        reference_audio: np.ndarray,
        estimate_audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]: ...


SymbolicMetricBuilder = Callable[["EvaluationSection"], SymbolicMetric | None]
AudioMetricBuilder = Callable[["EvaluationSection"], AudioMetric | None]

_SYMBOLIC_REGISTRY: dict[str, SymbolicMetricBuilder] = {}
_AUDIO_REGISTRY: dict[str, AudioMetricBuilder] = {}


def register_symbolic_metric(name: str) -> Callable[[SymbolicMetricBuilder], SymbolicMetricBuilder]:
    """Register a builder; it may return None to opt out for a given config."""

    def decorator(builder: SymbolicMetricBuilder) -> SymbolicMetricBuilder:
        _SYMBOLIC_REGISTRY[name] = builder
        return builder

    return decorator


def register_audio_metric(name: str) -> Callable[[AudioMetricBuilder], AudioMetricBuilder]:
    def decorator(builder: AudioMetricBuilder) -> AudioMetricBuilder:
        _AUDIO_REGISTRY[name] = builder
        return builder

    return decorator


def make_symbolic_metrics(section: "EvaluationSection") -> list[SymbolicMetric]:
    _ensure_builtins_registered()
    metrics = [builder(section) for builder in _SYMBOLIC_REGISTRY.values()]
    return [metric for metric in metrics if metric is not None]


def make_audio_metrics(section: "EvaluationSection") -> list[AudioMetric]:
    _ensure_builtins_registered()
    metrics = [builder(section) for builder in _AUDIO_REGISTRY.values()]
    return [metric for metric in metrics if metric is not None]


def evaluate_notes(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    metrics: Sequence[SymbolicMetric],
) -> dict[str, float]:
    """Run all metrics and flatten results into '<metric>.<key>' values."""
    results: dict[str, float] = {}
    for metric in metrics:
        for key, value in metric.compute(reference, estimate).items():
            results[f"{metric.name}.{key}"] = value
    return results


def _ensure_builtins_registered() -> None:
    from sonitra.evaluation import builtin_metrics  # noqa: F401
