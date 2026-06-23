from __future__ import annotations

import numpy as np

DEFAULT_FRAME_SIZE = 4096
DEFAULT_HOP_SIZE = 2048
DEFAULT_MAX_FRAMES = 4000


def chroma_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
) -> np.ndarray:
    """Compute a (frames, 12) chroma matrix from an audio array.

    Accepts mono (n,) or multi-channel (channels, n) audio; channels are mixed
    down. Each STFT bin's magnitude is accumulated into its nearest pitch
    class; rows are L2-normalised.
    """
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=0 if samples.shape[0] <= samples.shape[1] else 1)
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))

    n_frames = 1 + (samples.size - frame_size) // hop_size
    window = np.hanning(frame_size)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    valid = (freqs >= 27.5) & (freqs <= 8000.0)
    pitch_classes = np.zeros(freqs.size, dtype=int)
    # A440 is pitch class 9 in the conventional C=0 chroma ordering
    pitch_classes[valid] = (
        np.round(12.0 * np.log2(freqs[valid] / 440.0)).astype(int) + 9
    ) % 12

    chroma = np.zeros((n_frames, 12))
    for frame in range(n_frames):
        start = frame * hop_size
        spectrum = np.abs(np.fft.rfft(samples[start : start + frame_size] * window))
        np.add.at(chroma[frame], pitch_classes[valid], spectrum[valid])
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    return chroma / np.where(norms > 0, norms, 1.0)


def dtw_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Normalised DTW distance between two (frames, dims) feature matrices.

    Uses cosine distance as the local cost and steps (1,1), (1,0), (0,1); the
    accumulated cost is normalised by the warping path length so values are
    comparable across clip durations. Returns values in [0, 1] for
    L2-normalised non-negative features.
    """
    if features_a.shape[0] == 0 or features_b.shape[0] == 0:
        return float("nan")
    cost = 1.0 - features_a @ features_b.T
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)
    # Cells on anti-diagonal i+j depend only on the two previous diagonals,
    # so each diagonal can be updated as one vectorised operation.
    for diagonal in range(2, n + m + 1):
        i = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        if i.size == 0:
            continue
        j = diagonal - i
        candidates = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        best = np.argmin(candidates, axis=0)
        gather = np.arange(i.size)
        acc[i, j] = cost[i - 1, j - 1] + candidates[best, gather]
        origin_steps = np.stack([steps[i - 1, j - 1], steps[i - 1, j], steps[i, j - 1]])
        steps[i, j] = origin_steps[best, gather] + 1
    path_length = int(steps[n, m])
    return float(acc[n, m] / max(path_length, 1))


class DTWAudioMetric:
    """DTW-based audio similarity (Bradshaw et al. 2024).

    Compares two audio signals via chroma features and reports the
    path-normalised DTW cost; 0 means identical, larger values mean greater
    divergence. Long signals are decimated to at most ``max_frames`` feature
    frames to bound the quadratic alignment cost.
    """

    name = "dtw"

    def __init__(
        self,
        *,
        frame_size: int = DEFAULT_FRAME_SIZE,
        hop_size: int = DEFAULT_HOP_SIZE,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> None:
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.max_frames = int(max_frames)

    def compute(
        self,
        reference_audio: np.ndarray,
        estimate_audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        ref_features = self._features(reference_audio, sample_rate)
        est_features = self._features(estimate_audio, sample_rate)
        return {"distance": dtw_distance(ref_features, est_features)}

    def _features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        features = chroma_features(
            audio, sample_rate, frame_size=self.frame_size, hop_size=self.hop_size
        )
        if features.shape[0] > self.max_frames:
            stride = int(np.ceil(features.shape[0] / self.max_frames))
            features = features[::stride]
        return features
