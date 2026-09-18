# Model cards

Sonitra benchmarks audio-to-note transcription against a reference score; it does not train models. A transcription backend is the tool that turns audio into notes. A checkpoint is the saved weights for a trained model. This page documents every backend it ships: two trained models, Basic Pitch and TransKun, and two adapters, `precomputed` and `external_command`, that plug in transcriptions Sonitra did not produce itself.

Every backend takes audio and outputs MIDI notes. MIDI is a digital score format with pitch, start time, end time and loudness. Each model's card covers what it was built for, how it works, what data it saw, where it was published, its licence, and a compact spec table of the settings you need to run it.

Before comparing numbers across models or corpora, read these three notes.

MAESTRO overlap: TransKun was trained on MAESTRO V3, and Sonitra ships a `maestro-v3` corpus built from the same source. In `render_pipeline.input_type: midi` runs, Sonitra renders its own audio from the MIDI scores, so TransKun never hears an actual MAESTRO recording. The note sequences still overlap, because both come from the same scores. In `input_type: audio` runs, Sonitra plays the real MAESTRO recordings instead, so for TransKun that specific combination is training data at test time.

Input resolution: Basic Pitch resamples all input to 22.05 kHz. TransKun requires 44.1 kHz, loaded via `read_audio_resampled`. A side-by-side run does not control for input resolution, so a gap between the two models may reflect the resampling as well as the models themselves.

Velocity: velocity means how hard a note is hit, on a scale from 1 to 127. Basic Pitch's velocity is a rescaled model confidence score, not a measured loudness. TransKun predicts real velocity, trained against MAESTRO's recorded key-strike velocities. A side-by-side velocity comparison between the two models is not meaningful. See each model's Output fidelity note below for the mechanics.

### Applicability matrix

Every note Sonitra reads or produces, whether from the reference MIDI or a model's prediction, is reduced to four fields: `pitch`, `velocity`, `start_sec` and `duration_sec` (`src/sonitra/midi_reader.py`). MIDI channel is used only to track overlapping notes while a file is being read, then dropped. So on a multi-instrument corpus, both the reference and the prediction end up as a flat set of notes with no instrument label. Scoring compares those pitch sets; it does not transcribe each instrument separately.

Instrument scope has three separate axes. A model being polyphonic, meaning it can predict many notes sounding at once, does not mean it can tell instruments apart.

| Axis | Basic Pitch | TransKun |
|---|---|---|
| Trained timbre | Many instruments plus singing | Piano only |
| Polyphony | Yes. Upstream notes it works best on one instrument at a time | Yes |
| Instrument attribution | None: no per-note instrument label | None: no per-note instrument label |

The table below rates each shipped corpus (see [datasets.md](datasets.md)) against each model:

- `in-domain`: the model was built for that timbre.
- `out-of-domain but valid`: the model was not built for it, but still returns a scoreable, meaningful transcription.
- `category error`: the model's whole premise does not apply, so its output is not a meaningful measurement even though it runs without error.
- `train overlap`: the model saw this exact material while training.

| Corpus | Instrument(s) | Basic Pitch | TransKun |
|---|---|---|---|
| `maestro-v3` | Piano | in-domain | train overlap (see note below) |
| `bsed` | Orchestral (Beethoven symphony excerpts) | out-of-domain but valid | category error |
| `guitarset` | Acoustic guitar | in-domain | category error |
| `gaps` | Classical guitar | in-domain | category error |
| `musicnet` | Multi-instrument classical | out-of-domain but valid | category error |
| `e-gmd` | Drums | category error (see note below) | category error (see note below) |

Note on `maestro-v3`: both `input_type: midi` and `input_type: audio` runs draw their note sequences from the same MAESTRO scores, so both overlap TransKun's training data. `input_type: audio` runs go further and feed TransKun the actual MAESTRO recordings it trained on, not just the same notes.

Note on `e-gmd`: [datasets.md](datasets.md) states that Sonitra's transcription and scoring target pitched instruments, not drum hits, so E-GMD is not benchmarkable by either model. This is a tooling limit, not a difference between the two models.

TransKun does not refuse non-piano input. Point it at an orchestral or guitar recording and it returns plausible-looking, fully scoreable MIDI, complete with pitches, onsets, offsets and velocities. Nothing signals that the input was out of scope. A silent category error like this is worse than a crash, because a crash cannot be mistaken for a result.

### Basic Pitch

**Task.** Polyphonic note transcription and pitch estimation for many instruments. Polyphonic means many notes at once. The model outputs note events with pitch, start and end time, loudness and pitch bends. It works best on one instrument at a time.

**Architecture.** Lightweight convolutional neural network with harmonic stacking. The input is a constant Q transform. A constant Q transform is a pitch spaced spectrogram. The network predicts three heads at once: onsets, frames and contours. From those it builds notes and pitch bends. It is not a Transformer.

**Training data.** The original work trains on a mix of curated note level sets that include many instruments and singing. The exact split of training and held out sets for the released weight has changed since the paper. This page does not list a fixed count.

**Publication.** Bittner et al., ICASSP 2022 (Bittner, Bosch, Rubinstein, Meseguer-Brocal and Ewert, ICASSP 2022). The authors note the released code has been improved since the paper, so cite the code when you use its output.

**Licence.** Apache-2.0. Copyright 2022 Spotify AB. See the repository licence and the notice that the model data is subject to the same terms as the code distribution.

**Output fidelity.** Velocity is `round(float(amplitude) * 127)` (`src/sonitra/transcribe/basic_pitch.py`), where `amplitude` is the model's own note-activation confidence, not a measured loudness. Treat it as a rescaled confidence score, not a calibrated MIDI velocity. The model can also compute pitch bends, but `src/sonitra/midi_writer.py` writes no pitch-wheel MIDI messages, so bends never reach the output file. Setting `multiple_pitch_bends: true` only changes which overlapping same-pitch notes the model is allowed to emit; it does not add glissando curves, continuous slides between pitches, to the written MIDI.

| Property | Value |
|---|---|
| Sample rate | Any input rate. Resampled to 22.05 kHz before inference |
| Channels | Down mixed to mono at predict time |
| Model format | TensorFlow, also shipped as CoreML, TFLite and ONNX |
| Default model path | `ICASSP_2022_MODEL_PATH` from the `basic-pitch` package |
| Input length | Any length. The model windows internally |
| Pitch range | MIDI 21 to 108 with pitch bend support |
| Package | `basic-pitch` 0.4.x. Installed by default with Sonitra |
| Device config | `device: cpu` or `GPU:0` style. Passed to `tf.device` |

### TransKun

**Task.** Piano only transcription. The model turns a piano recording into MIDI notes with pitch, start, end and velocity. Velocity is loudness from 1 to 127.

**Architecture.** Transformer encoder plus a neural semi conditional random field. The encoder scores every possible time interval for being an event. The semi CRF layer decodes those scores into a set of non overlapping note intervals with dynamic programming. V2 replaces the original score module with a scaled inner product and simplifies the noise score.

**Training data.** MAESTRO V3, a set of 1,276 piano performances with aligned MIDI and audio. The default checkpoint shipped in the wheel is trained with data augmentation and without pedal extended note lengths. Other checkpoints are listed on the GitHub model cards table, with scores on MAESTRO, MAPS and SMD. Pedal extension means stretching each note to its sustain pedal release. The shipped weight skips that step.

**Publication.** Yan and Duan, ISMIR 2024 (Yan and Duan, ISMIR 2024) for V2. The V1 framework is Yan, Cwitkowitz and Duan, NeurIPS 2021 (Yan et al., NeurIPS 2021).

**Licence.** MIT. Wheel `transkun==2.0.1` bundles a 56 MB checkpoint at `transkun/pretrained/2.0.pt` with its conf at `pretrained/2.0.conf`. No download at runtime.

**Output fidelity.** TransKun predicts real velocity, trained against MAESTRO's recorded key-strike velocities, so its numbers mean something different from Basic Pitch's rescaled confidence score (see Velocity above); a side-by-side velocity comparison between the two is not meaningful. The checkpoint bundled in the wheel is trained without pedal-extended note lengths (see Training data above), so its note ends are literal key releases, not sustain-pedal releases. Reference MIDI that holds the sustain pedal down will disagree with TransKun systematically on note duration, which shows up in the onset+offset and key-overlap-ratio metrics (see [evaluation.md](evaluation.md)).

| Property | Value |
|---|---|
| Sample rate | 44.1 kHz. Loaded with `read_audio_resampled` via `pedalboard` `resampled_to(44100)` |
| Channels | Stereo or mono input. Passed as `(samples, channels)` to `model.transcribe` |
| Framework | PyTorch |
| Input length | Overlapping windows. Sonitra keeps the bundled conf as truth: 16 s window with 8 s hop when fields are `null`. CLI help shows 20 s and 10 s as its own defaults |
| Pitch range | MIDI 21 to 108. Raw output also contains -64 and -67 for sustain and soft pedal, filtered before `make_note` |
| Package | `transkun==2.0.1` under the `transkun` extra. `pip install sonitra[transkun]` |
| Device config | `cpu`, `cuda`, `cuda:1`, `mps` or `GPU:0` style. Sonitra translates `GPU:0` to `cuda:0`. A `cuda` request with no CUDA raises |
| Checkpoint location | `importlib.resources` under `transkun/pretrained/`. Override with `weights_path` and `conf_path` |

### precomputed

**Task.** Looks up a MIDI file that already exists on disk for a given audio file, matched by file stem. It is the adapter for black-box commercial transcription tools, such as exported output from klang.io, Moises or AnthemScore, that Sonitra cannot run itself.

**Configuration.** Set `midi_dir` to the folder holding the exported MIDI. `extensions` (default `.mid`, `.midi`) lists the file suffixes to try, in order.

**Provenance.** Provenance means the record of where a result came from, such as which model version or checkpoint produced it. Sonitra records no provenance metadata at all for a `precomputed` result: no model name, no version, no checkpoint identity. Anyone publishing numbers from this backend needs to record that information themselves, outside Sonitra.

### external_command

**Task.** Runs any command-line transcription tool that reads an audio file and writes a MIDI file. The `command` template must contain `{input}` and `{output}` placeholders; Sonitra substitutes real paths and runs the tool as a subprocess.

**Configuration.** `command` is required. `output_extension` defaults to `.mid`. `timeout_sec` defaults to 600.

**Provenance.** Sonitra records only the command string in the result's metadata (`src/sonitra/transcribe/external_command.py`). It cannot record a model version or checkpoint identity for whatever the command actually runs underneath, the same provenance gap as `precomputed`. That matters for anyone publishing numbers produced this way.

### Measured baselines

The table below is generated by `scripts/export_model_baselines.py` from real benchmark runs. Its columns are drawn from three of Sonitra's four metric families: note-level, frame-level and expressive (see [evaluation.md](evaluation.md) for what each one measures). The fourth family, audio DTW, is optional and not shown here; DTW means dynamic time warping, a way to line up two audio clips in time and measure the gap between them. The `n` column counts files that scored successfully; when some failed, it reads `succeeded/total`.

These are baseline-condition results only, from benchmark runs on the maintainer's machines. Scores depend on the corpus, the input mode and the render settings, not on the model alone, so no single number here should be quoted out of that context.

The `corpus/` directory is not tracked in git, so you cannot regenerate this table from a fresh checkout. Each run's own `summary.json` records its host and timing, for anyone who needs that detail.

TransKun is absent from the table when it has no qualifying benchmark run yet. That is not a comment on how it performed; the generated block below names it explicitly when this is the case.

Velocity correlation is undefined for GuitarSet, for every model. `scripts/guitarset_jams_to_midi.py` writes a fixed velocity of 100 for every reference note (see [datasets.md](datasets.md)), so the reference carries no velocity variation to correlate against.

Refresh the table with:

```bash
python scripts/export_model_baselines.py --stdout   # preview
python scripts/export_model_baselines.py            # rewrite the generated block
python scripts/export_model_baselines.py --check    # verify the table is current
```

`--check` is a local maintainer command. It cannot run in CI, because CI has no `corpus/`.

<!-- BEGIN GENERATED: baselines -->
<!-- Generated by scripts/export_model_baselines.py - do not edit by hand -->

| Corpus | Input | Model | n | onset F1 | +offset F1 | +vel F1 | vel corr | frame F1 |
|---|---|---|---|---|---|---|---|---|
| bsed | audio | basic_pitch | 100 | 0.056 | 0.016 | 0.011 | -0.092 | 0.300 |
| bsed | midi | basic_pitch | 20 | 0.641 | 0.333 | 0.206 | -0.107 | 0.676 |
| guitarset | audio | basic_pitch | 720 | 0.772 | 0.546 | 0.546 | — | 0.827 |
| guitarset | midi | basic_pitch | 360 | 0.860 | 0.298 | 0.298 | — | 0.787 |
| maestro-v3 | audio | basic_pitch | 1276 | 0.662 | 0.109 | 0.050 | 0.031 | 0.535 |
| maestro-v3 | midi | basic_pitch | 1276 | 0.817 | 0.136 | 0.059 | 0.055 | 0.661 |

_`transkun` is omitted: no qualifying benchmark run. Its only baseline data is `test/transkun_baseline` (n=1), too little to compare fairly._

_Baseline condition only, from 6 run(s) under `corpus/`; newest run 2026-09-12. Each run's own `summary.json` carries its host and timing._
<!-- END GENERATED: baselines -->

---
[← Back to README](../README.md)
