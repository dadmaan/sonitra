from __future__ import annotations

import math
from typing import Sequence

from sonitra.evaluation.note_metrics import precision_recall_f1
from sonitra.evaluation.types import NoteEvent

DEFAULT_HOP_SEC = 0.01


def rasterise(notes: Sequence[NoteEvent], *, hop_sec: float = DEFAULT_HOP_SEC) -> set[tuple[int, int]]:
    """Rasterise notes into a set of active (pitch, frame_index) cells."""
    active: set[tuple[int, int]] = set()
    for note in notes:
        first = int(math.floor(note.onset_sec / hop_sec))
        last = int(math.ceil(note.offset_sec / hop_sec))
        if last <= first:
            last = first + 1
        for frame in range(first, last):
            active.add((note.pitch, frame))
    return active


class FrameMetrics:
    """Frame-level precision/recall/F1 over binarised piano rolls."""

    name = "frame"

    def __init__(self, *, hop_sec: float = DEFAULT_HOP_SEC) -> None:
        self.hop_sec = float(hop_sec)

    def compute(self, reference: Sequence[NoteEvent], estimate: Sequence[NoteEvent]) -> dict[str, float]:
        ref_cells = rasterise(reference, hop_sec=self.hop_sec)
        est_cells = rasterise(estimate, hop_sec=self.hop_sec)
        matched = len(ref_cells & est_cells)
        precision, recall, f1 = precision_recall_f1(matched, len(est_cells), len(ref_cells))
        return {"precision": precision, "recall": recall, "f1": f1}
