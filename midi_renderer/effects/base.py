from __future__ import annotations

from typing import Protocol

import numpy as np


class EffectsChain(Protocol):
    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...
