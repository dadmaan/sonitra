# Evaluation metrics

Sonitra checks how well a transcription matches the original score. The table below lists the four groups of checks you get. Note means a single musical note. Onset means the moment a note starts. Offset means the moment a note ends. Velocity means how hard a note is hit.

| Family | What is measured |
|---|---|
| Note-level | Onset F1 (±50 ms), onset+offset F1, onset+offset+velocity F1 |
| Frame-level | Precision/recall/F1 over 10 ms piano-roll frames |
| Expressive | Onset MAE/bias, IOI correlation, key-overlap-ratio correlation, velocity correlation, windowed pitch-class harmony similarity |
| Audio (optional) | Path-normalised DTW distance over chroma features between rendered audio and re-synthesised transcription |

Sonitra computes these scores with NumPy and SciPy, two common Python math libraries. Note matching follows mir_eval, a standard music-evaluation library. Mir_eval uses bipartite matching, which means it pairs each predicted note with at most one true note to find the best fit. If a score has no clear value, for example a correlation based on too few matched notes, Sonitra reports `NaN` (not a number) and leaves it out of averages.

See [research.md](research.md) for background reading on why Sonitra uses these checks.

---
[← Back to README](../README.md)
