# Evaluation metrics

| Family | What is measured |
|---|---|
| Note-level | Onset F1 (±50 ms), onset+offset F1, onset+offset+velocity F1 |
| Frame-level | Precision/recall/F1 over 10 ms piano-roll frames |
| Expressive | Onset MAE/bias, IOI correlation, key-overlap-ratio correlation, velocity correlation, windowed pitch-class harmony similarity |
| Audio (optional) | Path-normalised DTW distance over chroma features between rendered audio and re-synthesised transcription |

Metrics are implemented in NumPy/SciPy with mir_eval-compatible bipartite matching semantics. Undefined values (e.g. correlations over too few matched notes) are reported as `NaN` and skipped in aggregation.

See [research.md](research.md) for the literature survey backing these metric choices.

---
[← Back to README](../README.md)
