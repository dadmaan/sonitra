# Numeric reproducibility

Sonitra produces comparable numbers; this note records where that comparability is and is not guaranteed.

## How equivalence is measured

`compare_variants.py` renders each file once, transcribes the same audio with variants A and B, and reports A-vs-B directly plus each variant against the reference MIDI as ΔF1 = |F1_A − F1_B|. F1 is a 0-to-1 score for how closely predicted notes match a reference, where 1 is a perfect match. Sonitra computes it four ways: note onset; onset+offset; onset+offset+velocity; and frame by frame, giving the keys `note.onset_f1`, `note.onset_offset_f1`, `note.onset_offset_velocity_f1` and `frame.f1`. The control arm (the same variant run twice) must report exactly zero; that zero is the noise floor. Any non-zero ΔF1 is then judged against the benchmark `summary.json` degradation table (materiality).

## Scope

Measured: `basic_pitch` and `transkun`, both neural backends, and separately `hft_transformer`. These are not one experiment. Each backend was measured against its own bounds, so read the arm table together with the conditions named beside it and do not carry one backend's result across to another.

All Phase-2 and Phase-3e numbers below use the `bsed` dataset at `--limit 20` with `batch_size=1`, except the timing profile, which uses a 363 s MAESTRO render. Region of validity, stated plainly because it is easy to over-read: the precursor measurement was `basic_pitch`-only, and arm 2b is what confirms the finding on `transkun`.

Measured for `hft_transformer`, to different bounds. On CPU at `batch_size=1` the vendored network and note decoder are bit-for-bit identical to the upstream reference implementation, checked across 43 windows and both window modes, so the port itself adds no numeric change. CPU against CUDA diverges by up to 3.3e-04 on the velocity logits, which changes 70–75 of roughly 75 note-list entries in the last bits. CPU `batch_size=4` against `batch_size=1` was bitwise identical on that machine. It has not been run through the `bsed` arm design below, and the `transkun` rows there do not cover it.

Measured: nothing else. `precomputed` and `external_command` are not neural (they replay a stored file or shell out to a command), so device and numeric-mode settings do not apply to them. Do not extend these numbers to any backend that was not run.

## The pedal-blindness caveat

Any onset+offset or velocity-weighted number for `hft_transformer` is a lower bound, and the reason is Sonitra's reference side rather than the model. That model's training references are pedal-extended: the reference builder forces note-off at the CC64 sustain-pedal release, so notes ring until the pedal lifts. `src/sonitra/midi_reader.py` has no `control_change` branch, so it cannot reproduce that convention: reference durations here came out 22.2 % shorter than the model's own training references, +993.5 s of reference note length measured against +180.4 s. `note.onset_f1` is convention-neutral and is the number to compare; `note.onset_offset_f1` and `note.onset_offset_velocity_f1` mix model quality with a reference-construction difference. This affects the scoring target, not the transcription: it is a property of the MIDI Sonitra reads as truth, and it applies to any model scored against that reference.

## Where basic_pitch time goes

Steady-state per 363 s file (222 windows): 2.85 s inference (56%), 1.89 s post-processing, 0.33 s load; 5.07 s CPU vs 3.51 s GPU per file. Single-file profiles mislead: librosa's ~11 s per-process warmup gets charged to the load phase, reading as I/O-bound. Batching windows gives 5.53x CPU inference (1.85x end-to-end); batched CPU (2.74 s) beats unbatched GPU (3.51 s).

## What diverges, and the switch

Precursor (basic_pitch only): CPU vs GPU moved one note (1781 vs 1780 events); GPU batch-size shifts reach 1.29e-02 against 8.05e-07 on CPU. A note activation is a probability, and onset detection cuts off at 0.5, so a shift that large can flip a note.

Phase-2 arms, each the same config pair over `bsed` (`--limit 20`, `batch_size=1`):

| Arm | Comparison | Files changed | Largest ΔF1 (four keys) | Note-count Δ |
| --- | --- | --- | --- | --- |
| 2d | same variant twice (noise floor, n=19) | 0/19 | exactly 0, all keys | 0 |
| 2a | `basic_pitch` cpu vs cuda | 1/20 | onset 0.00455, frame 0.00425, +offset 0.00092, velocity 9e-5 | max 1, mean 0.05 |
| 2b | `transkun` cpu vs cuda | 4/20 | frame 0.00689, onset 0.00536, +offset 0.00560, velocity 0.00566 | max 1, mean 0.15 |
| 2c | `basic_pitch` cpu vs itself, `TF_ENABLE_ONEDNN_OPTS=1` vs `=0` | 1/19 | frame 8.8e-5 only; note-level and note-count 0 | 0 |

2d is the noise floor: identical config twice is exactly zero on all four keys, so any non-zero arm delta is real and not run-to-run noise.

2b is the generalisation. `transkun` is a Transformer with a semi-CRF decoder (semi-CRF: a sequence decoder that picks note boundaries jointly, rather than thresholding each frame alone), structurally different from `basic_pitch`'s small CNN. It diverges on device too, so device divergence is not a `basic_pitch`/CNN artefact.

2c is CPU against itself under a documented TensorFlow flag (`TF_ENABLE_ONEDNN_OPTS`; oneDNN is TensorFlow's CPU maths library, and the flag switches its custom kernels on or off). Note-level and note-count differences are exactly zero; only one file's `frame.f1` differs, by 8.8e-5. A same-environment two-process control is exactly zero, so that delta can be attributed to the flag. CPU self-divergence is real but limited to frame metrics at 8.8e-5.

Materiality. The largest real condition effect on the same four keys in an existing `summary.json` degradation table is |Δnote.onset_f1| 0.7922 (`corpus/guitarset/benchmark/guitar_only_MIDI`, condition `gain=35_cab=off_rev=0.9_dly=on`, basic_pitch); the largest maestro-v3 effect is 0.7044. The device-induced maximum of 0.0069 is about 115x and 102x smaller respectively. That makes device divergence a documentation note, not a disqualifying effect.

`transcription.numeric_mode` (`off`/`warn`/`strict`, default `off`) is process-global (it configures the whole program, not one transcriber), stays in the run fingerprint (the hash that tags a set of results, so changing the mode starts a new set), and is recorded per-row in transcriber metadata; `warn` falls back where `strict` raises. `config/benchmark/methods/numerics_check.yaml` is a transcription-only methods preset, not a model comparison. The preset is top-level-only: the runner builds transcribers once from the base config, so per-condition numeric overrides are recorded but not applied. That is a methods artefact, and the reason the header tells you to run each mode as its own top-level run.

Phase-3e acceptance re-ran 2a and 2b with `numeric_mode: strict` on both sides. Both arms left 0/20 files changed, all four keys exactly 0, so the switch removes the divergence without crashing. Median per-file runtime with model-load warmup excluded is unchanged: basic_pitch cpu 0.200 → 0.199 s and cuda 0.103 → 0.100 s, transkun cpu 7.71 → 7.44 s, transkun cuda 1.315 → 1.377 s (+4.7%). No material cost.

## Audio loaders

`basic_pitch` and `transkun` do not share a loader, and are not intended to. `basic_pitch` uses `read_audio_basic_pitch` (`soundfile` read → mean-to-mono → `soxr` HQ resample), which reproduces its old `librosa.load` input bit-for-bit without librosa's per-process numba warmup. `transkun` keeps `read_audio_resampled` (`pedalboard`), which handles wav, flac and mp3 without ffmpeg. A single shared loader was considered and not adopted: on a real mp3 the decode length differs between pedalboard and soundfile, so unifying the two paths would change transkun's input. `docs/adding-a-transcriber.md` documents the transkun loader.

## Known limitations

Provenance (the record of how a result was produced) does not reach the summary. `transcriber_metadata` (`device`, `requested_device`, `device_available`, `numeric_mode`) is written to `benchmark_results.jsonl` only; it never reaches `summary.json`, because `summarise()` is metrics-only by design (`src/sonitra/benchmark/results.py:124-163`). `summary.json`'s `timing.host.gpu` records hardware present, not the device used, so a CPU run on a GPU host still reports a GPU. `scripts/export_model_baselines.py` reads only `render_pipeline.input_type` from `config.yaml` and cannot show a device or numeric-mode column today. This is a known limitation; surfacing it per baseline row is deferred to a separate ticket.
