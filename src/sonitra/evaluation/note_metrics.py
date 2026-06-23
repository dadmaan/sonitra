from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from sonitra.evaluation.types import NoteEvent

DEFAULT_ONSET_TOLERANCE_SEC = 0.05
DEFAULT_OFFSET_RATIO = 0.2
DEFAULT_OFFSET_MIN_TOLERANCE_SEC = 0.05
DEFAULT_VELOCITY_TOLERANCE = 0.1


def match_notes(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    *,
    onset_tolerance_sec: float = DEFAULT_ONSET_TOLERANCE_SEC,
    with_offset: bool = False,
    offset_ratio: float = DEFAULT_OFFSET_RATIO,
    offset_min_tolerance_sec: float = DEFAULT_OFFSET_MIN_TOLERANCE_SEC,
) -> list[tuple[int, int]]:
    """Maximum bipartite matching between reference and estimated notes.

    Follows the mir_eval.transcription conventions: a candidate pair requires
    equal pitch and an onset within ``onset_tolerance_sec``; with
    ``with_offset`` the offsets must additionally agree within
    ``max(offset_min_tolerance_sec, offset_ratio * reference duration)``.
    Returns (reference_index, estimate_index) pairs.
    """
    rows: list[int] = []
    cols: list[int] = []
    for i, ref in enumerate(reference):
        for j, est in enumerate(estimate):
            if ref.pitch != est.pitch:
                continue
            if abs(ref.onset_sec - est.onset_sec) > onset_tolerance_sec:
                continue
            if with_offset:
                tolerance = max(offset_min_tolerance_sec, offset_ratio * ref.duration_sec)
                if abs(ref.offset_sec - est.offset_sec) > tolerance:
                    continue
            rows.append(i)
            cols.append(j)
    if not rows:
        return []
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(reference), len(estimate)),
    )
    assignment = maximum_bipartite_matching(graph, perm_type="column")
    return [(i, int(j)) for i, j in enumerate(assignment) if j >= 0]


def match_notes_with_velocity(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    *,
    velocity_tolerance: float = DEFAULT_VELOCITY_TOLERANCE,
    **match_kwargs,
) -> list[tuple[int, int]]:
    """Velocity-aware matching following mir_eval.transcription_velocity.

    Notes are first matched on onset/offset criteria; estimated velocities are
    then rescaled onto the reference scale by least squares, and matches whose
    rescaled velocity deviates by more than ``velocity_tolerance`` (relative to
    the maximum reference velocity) are dropped.
    """
    pairs = match_notes(reference, estimate, **match_kwargs)
    if not pairs:
        return []
    ref_vel = np.array([reference[i].velocity for i, _ in pairs], dtype=float)
    est_vel = np.array([estimate[j].velocity for _, j in pairs], dtype=float)
    max_ref = ref_vel.max()
    if max_ref <= 0:
        return pairs
    ref_norm = ref_vel / max_ref
    if np.ptp(est_vel) > 0:
        slope, intercept = np.polyfit(est_vel, ref_norm, 1)
        est_norm = slope * est_vel + intercept
    else:
        est_norm = np.full_like(est_vel, float(ref_norm.mean()))
    keep = np.abs(est_norm - ref_norm) <= velocity_tolerance
    return [pair for pair, ok in zip(pairs, keep) if ok]


def precision_recall_f1(n_matched: int, n_estimate: int, n_reference: int) -> tuple[float, float, float]:
    precision = n_matched / n_estimate if n_estimate else 0.0
    recall = n_matched / n_reference if n_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class NoteMetrics:
    """Note-level onset / onset+offset / onset+offset+velocity P/R/F1."""

    name = "note"

    def __init__(
        self,
        *,
        onset_tolerance_sec: float = DEFAULT_ONSET_TOLERANCE_SEC,
        offset_ratio: float = DEFAULT_OFFSET_RATIO,
        offset_min_tolerance_sec: float = DEFAULT_OFFSET_MIN_TOLERANCE_SEC,
        velocity_tolerance: float = DEFAULT_VELOCITY_TOLERANCE,
    ) -> None:
        self.onset_tolerance_sec = float(onset_tolerance_sec)
        self.offset_ratio = float(offset_ratio)
        self.offset_min_tolerance_sec = float(offset_min_tolerance_sec)
        self.velocity_tolerance = float(velocity_tolerance)

    def compute(self, reference: Sequence[NoteEvent], estimate: Sequence[NoteEvent]) -> dict[str, float]:
        results: dict[str, float] = {}
        onset_pairs = match_notes(
            reference,
            estimate,
            onset_tolerance_sec=self.onset_tolerance_sec,
        )
        offset_pairs = match_notes(
            reference,
            estimate,
            onset_tolerance_sec=self.onset_tolerance_sec,
            with_offset=True,
            offset_ratio=self.offset_ratio,
            offset_min_tolerance_sec=self.offset_min_tolerance_sec,
        )
        velocity_pairs = match_notes_with_velocity(
            reference,
            estimate,
            velocity_tolerance=self.velocity_tolerance,
            onset_tolerance_sec=self.onset_tolerance_sec,
            with_offset=True,
            offset_ratio=self.offset_ratio,
            offset_min_tolerance_sec=self.offset_min_tolerance_sec,
        )
        for label, pairs in (
            ("onset", onset_pairs),
            ("onset_offset", offset_pairs),
            ("onset_offset_velocity", velocity_pairs),
        ):
            precision, recall, f1 = precision_recall_f1(len(pairs), len(estimate), len(reference))
            results[f"{label}_precision"] = precision
            results[f"{label}_recall"] = recall
            results[f"{label}_f1"] = f1
        return results
