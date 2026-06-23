from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from sonitra.evaluation.note_metrics import DEFAULT_ONSET_TOLERANCE_SEC, match_notes
from sonitra.evaluation.types import NoteEvent

DEFAULT_HARMONY_WINDOW_SEC = 2.0


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pitch_class_histograms(
    notes: Sequence[NoteEvent], *, window_sec: float, n_windows: int
) -> np.ndarray:
    """Duration-weighted pitch-class histogram per analysis window."""
    histograms = np.zeros((n_windows, 12))
    for note in notes:
        first = int(math.floor(note.onset_sec / window_sec))
        last = int(math.ceil(note.offset_sec / window_sec))
        for window in range(max(0, first), min(n_windows, max(last, first + 1))):
            start = window * window_sec
            overlap = min(note.offset_sec, start + window_sec) - max(note.onset_sec, start)
            histograms[window, note.pitch % 12] += max(0.0, overlap)
    return histograms


class ExpressiveMetrics:
    """Musically informed metrics inspired by mpteval (Hu et al. 2024).

    All correlation metrics are computed over note pairs matched on onset and
    pitch, ordered by reference onset:

    - ``onset_mae`` / ``onset_bias``: absolute and signed mean onset error.
    - ``ioi_corr``: Pearson correlation of consecutive inter-onset intervals
      (timing fidelity).
    - ``kor_corr``: Pearson correlation of key-overlap ratios, the articulation
      measure (offset-to-next-onset overlap divided by the IOI).
    - ``velocity_corr``: Pearson correlation of matched velocities (dynamics).
    - ``harmony_sim``: mean cosine similarity of windowed duration-weighted
      pitch-class histograms, a tractable proxy for tonal-cloud measures.

    Metrics that are undefined for the given input (too few matches, zero
    variance) are reported as NaN and skipped during aggregation.
    """

    name = "expressive"

    def __init__(
        self,
        *,
        onset_tolerance_sec: float = DEFAULT_ONSET_TOLERANCE_SEC,
        harmony_window_sec: float = DEFAULT_HARMONY_WINDOW_SEC,
    ) -> None:
        self.onset_tolerance_sec = float(onset_tolerance_sec)
        self.harmony_window_sec = float(harmony_window_sec)

    def compute(self, reference: Sequence[NoteEvent], estimate: Sequence[NoteEvent]) -> dict[str, float]:
        pairs = match_notes(reference, estimate, onset_tolerance_sec=self.onset_tolerance_sec)
        pairs.sort(key=lambda pair: reference[pair[0]].onset_sec)
        ref_matched = [reference[i] for i, _ in pairs]
        est_matched = [estimate[j] for _, j in pairs]

        results: dict[str, float] = {
            "onset_mae": float("nan"),
            "onset_bias": float("nan"),
            "ioi_corr": float("nan"),
            "kor_corr": float("nan"),
            "velocity_corr": float("nan"),
            "harmony_sim": self._harmony_similarity(reference, estimate),
        }
        if not pairs:
            return results

        onset_errors = np.array(
            [est.onset_sec - ref.onset_sec for ref, est in zip(ref_matched, est_matched)]
        )
        results["onset_mae"] = float(np.mean(np.abs(onset_errors)))
        results["onset_bias"] = float(np.mean(onset_errors))

        ref_onsets = np.array([note.onset_sec for note in ref_matched])
        est_onsets = np.array([note.onset_sec for note in est_matched])
        results["ioi_corr"] = _pearson(np.diff(ref_onsets), np.diff(est_onsets))
        results["kor_corr"] = _pearson(
            _key_overlap_ratios(ref_matched), _key_overlap_ratios(est_matched)
        )
        results["velocity_corr"] = _pearson(
            np.array([note.velocity for note in ref_matched], dtype=float),
            np.array([note.velocity for note in est_matched], dtype=float),
        )
        return results

    def _harmony_similarity(
        self, reference: Sequence[NoteEvent], estimate: Sequence[NoteEvent]
    ) -> float:
        end = max(
            [note.offset_sec for note in reference] + [note.offset_sec for note in estimate],
            default=0.0,
        )
        if end <= 0.0:
            return float("nan")
        n_windows = int(math.ceil(end / self.harmony_window_sec))
        ref_hist = pitch_class_histograms(
            reference, window_sec=self.harmony_window_sec, n_windows=n_windows
        )
        est_hist = pitch_class_histograms(
            estimate, window_sec=self.harmony_window_sec, n_windows=n_windows
        )
        ref_norm = np.linalg.norm(ref_hist, axis=1)
        est_norm = np.linalg.norm(est_hist, axis=1)
        active = (ref_norm > 0) | (est_norm > 0)
        if not np.any(active):
            return float("nan")
        denominator = np.where(ref_norm * est_norm > 0, ref_norm * est_norm, 1.0)
        cosine = np.sum(ref_hist * est_hist, axis=1) / denominator
        cosine = np.where((ref_norm > 0) & (est_norm > 0), cosine, 0.0)
        return float(np.mean(cosine[active]))


def _key_overlap_ratios(notes: Sequence[NoteEvent]) -> np.ndarray:
    """Key-overlap ratio per consecutive note: (offset_i - onset_{i+1}) / IOI_i."""
    if len(notes) < 2:
        return np.array([])
    ratios = []
    for current, following in zip(notes, notes[1:]):
        ioi = following.onset_sec - current.onset_sec
        if ioi <= 0:
            ratios.append(0.0)
        else:
            ratios.append((current.offset_sec - following.onset_sec) / ioi)
    return np.array(ratios)
