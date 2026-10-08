from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EffectsChain(Protocol):
    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

    def __len__(self) -> int: ...
