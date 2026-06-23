from __future__ import annotations

from typing import TYPE_CHECKING

from sonitra.evaluation.dtw_metric import DTWAudioMetric
from sonitra.evaluation.expressive_metrics import ExpressiveMetrics
from sonitra.evaluation.frame_metrics import FrameMetrics
from sonitra.evaluation.note_metrics import NoteMetrics
from sonitra.evaluation.protocol import register_audio_metric, register_symbolic_metric

if TYPE_CHECKING:
    from sonitra.config import EvaluationSection


@register_symbolic_metric("note")
def _build_note_metrics(section: "EvaluationSection") -> NoteMetrics | None:
    if not section.note_metrics.enabled:
        return None
    return NoteMetrics(
        onset_tolerance_sec=section.note_metrics.onset_tolerance_sec,
        offset_ratio=section.note_metrics.offset_ratio,
        offset_min_tolerance_sec=section.note_metrics.offset_min_tolerance_sec,
        velocity_tolerance=section.note_metrics.velocity_tolerance,
    )


@register_symbolic_metric("frame")
def _build_frame_metrics(section: "EvaluationSection") -> FrameMetrics | None:
    if not section.frame_metrics.enabled:
        return None
    return FrameMetrics(hop_sec=section.frame_metrics.hop_sec)


@register_symbolic_metric("expressive")
def _build_expressive_metrics(section: "EvaluationSection") -> ExpressiveMetrics | None:
    if not section.expressive_metrics.enabled:
        return None
    return ExpressiveMetrics(
        onset_tolerance_sec=section.note_metrics.onset_tolerance_sec,
        harmony_window_sec=section.expressive_metrics.harmony_window_sec,
    )


@register_audio_metric("dtw")
def _build_dtw_metric(section: "EvaluationSection") -> DTWAudioMetric | None:
    if not section.dtw.enabled:
        return None
    return DTWAudioMetric(
        frame_size=section.dtw.frame_size,
        hop_size=section.dtw.hop_size,
        max_frames=section.dtw.max_frames,
    )
