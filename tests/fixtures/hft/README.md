# hFT-Transformer parity fixtures

Reference outputs recorded from the **unmodified** upstream code of
[hFT-Transformer](https://github.com/sony/hFT-Transformer) (ISMIR 2023, MIT),
commit `71a2ee06e9ced1ea24673c95ee0acded2fc98d04`, with `torch 2.12.1+cu130`
on CPU at `batch_size=1`. Upstream was imported and executed as published — no
source edits, no stubs in the code under test — so these files are a parity
reference rather than a snapshot of our own output.

## Contents

| Files | Contents |
| --- | --- |
| `transcript_*.npy` | `AMT.transcript` outputs for both decoder heads: `onset`, `offset`, `mpe` as `float32`, `velocity` as `int8`, each `[256, 88]`. |
| `mpe2note_*.npz` | `AMT.mpe2note` note lists, one archive per edge case, as parallel arrays `pitch` (`int16`), `onset`/`offset` (`float32`), `velocity` (`int8`). |
| `MANIFEST.json` | Sizes, shapes, the reason each edge case exists, and the generator constants below. |

## Regenerating the inputs

No input array is stored. `transcript_*.npy` comes from a 256-frame log-mel block
built from the `generator` block of `MANIFEST.json`:

```python
rng = numpy.random.default_rng(20261003)          # random_seed
values = rng.normal(-7.136, 5.071, size=(256, 256)).astype("float32")  # centre, spread, frames, n_bins
features = numpy.clip(values, -18.42068099975586, None).astype("float32")  # min_value
```

The centre and spread are the measured mean and standard deviation of a real
log-mel block, so the activations land in the model's operating range rather than
below every threshold. `min_value` is `float32(log(1e-8))`, the padding value
`transcript` writes around each window. `mpe2note_*.npz` are driven by
thresholds `0.5` for onset, offset and MPE.

The edge cases in `mpe2note_*.npz` each pin one branch of the note detector:

- `plateau_above_threshold` — a tie on every frame, so the scans reach both ends of
  the array and the onset/offset interpolation takes its equality branch.
- `velocity_zero_ignored` — velocity 0 at the onset frame; no note may survive.
- `onset_at_frame_0` — a peak on frame 0, where no interpolation is possible.
- `same_pitch_overlap_trim` — three onsets on one pitch; each earlier note must be
  shortened to the next onset.
- `note_to_last_frame` — a peak on the final frame, where the offset falls back to
  `(frames - 1) * hop` and may land past the audio end.
- `dense_random_onsets` — local maxima on all 88 pitches with zero velocities
  interleaved, which exercises the velocity drop at scale.

## Using them

`tests/test_hft_vendor.py` compares the vendored implementation against these
files. Matching output byte for byte at `batch_size=1` on CPU is the parity
condition; on CUDA, and for `batch_size > 1`, floating-point reduction order
differs, so compare with a tolerance instead.