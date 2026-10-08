# Model cards

Sonitra benchmarks audio-to-note transcription against a reference score; it does not train models. A transcription backend is the tool that turns audio into notes. A checkpoint is the saved weights for a trained model. This page documents every backend it ships: three trained models, Basic Pitch, TransKun and hFT-Transformer, and two adapters, `precomputed` and `external_command`, that plug in transcriptions Sonitra did not produce itself.

Every backend takes audio and outputs MIDI notes. MIDI is a digital score format with pitch, start time, end time and loudness. Each model's card covers what it was built for, how it works, what data it saw, where it was published, its licence, and a compact spec table of the settings you need to run it.

Before comparing numbers across models or corpora, read these three notes.

MAESTRO overlap: TransKun was trained on MAESTRO V3, and Sonitra ships a `maestro-v3` corpus built from the same source. In `render_pipeline.input_type: midi` runs, Sonitra renders its own audio from the MIDI scores, so TransKun never hears an actual MAESTRO recording. The note sequences still overlap, because both come from the same scores. In `input_type: audio` runs, Sonitra plays the real MAESTRO recordings instead, so for TransKun that specific combination is training data at test time. hFT-Transformer is trained on MAESTRO V3 too, so the same two cases apply to it: both piano models have train overlap on this corpus, and neither has ever heard Sonitra's own renders.

Input resolution: Basic Pitch resamples all input to 22.05 kHz. TransKun requires 44.1 kHz, loaded via `read_audio_resampled`. hFT-Transformer's feature extractor is pinned to 16 kHz. A side-by-side run does not control for input resolution, so a gap between the models may reflect the resampling as well as the models themselves.

Velocity: velocity means how hard a note is hit, on a scale from 1 to 127. Basic Pitch's velocity is a rescaled model confidence score, not a measured loudness. TransKun predicts real velocity, trained against MAESTRO's recorded key-strike velocities. A side-by-side velocity comparison between Basic Pitch and either piano model is not meaningful; between TransKun and hFT-Transformer it is, because both predict real velocity against the same source. See each model's Output fidelity note below for the mechanics.

### Applicability matrix

Every note Sonitra reads or produces, whether from the reference MIDI or a model's prediction, is reduced to four fields: `pitch`, `velocity`, `start_sec` and `duration_sec` (`src/sonitra/midi_reader.py`). MIDI channel is used only to track overlapping notes while a file is being read, then dropped. So on a multi-instrument corpus, both the reference and the prediction end up as a flat set of notes with no instrument label. Scoring compares those pitch sets; it does not transcribe each instrument separately.

Instrument scope has three separate axes. A model being polyphonic, meaning it can predict many notes sounding at once, does not mean it can tell instruments apart.

| Axis | Basic Pitch | TransKun | hFT-Transformer |
|---|---|---|---|
| Trained timbre | Many instruments plus singing | Piano only | Piano only |
| Polyphony | Yes. Upstream notes it works best on one instrument at a time | Yes | Yes |
| Instrument attribution | None: no per-note instrument label | None: no per-note instrument label | None: no per-note instrument label |

The table below rates each shipped corpus (see [datasets.md](datasets.md)) against each model:

- `in-domain`: the model was built for that timbre.
- `out-of-domain but valid`: the model was not built for it, but still returns a scoreable, meaningful transcription.
- `category error`: the model's whole premise does not apply, so its output is not a meaningful measurement even though it runs without error.
- `train overlap`: the model saw this exact material while training.

| Corpus | Instrument(s) | Basic Pitch | TransKun | hFT-Transformer |
|---|---|---|---|---|
| `maestro-v3` | Piano | in-domain | train overlap (see note below) | train overlap (see note below) |
| `bsed` | Orchestral (Beethoven symphony excerpts) | out-of-domain but valid | category error | category error |
| `guitarset` | Acoustic guitar | in-domain | category error | category error |
| `gaps` | Classical guitar | in-domain | category error | category error |
| `musicnet` | Multi-instrument classical | out-of-domain but valid | category error | category error |
| `e-gmd` | Drums | category error (see note below) | category error (see note below) | category error (see note below) |

Note on `maestro-v3`: both `input_type: midi` and `input_type: audio` runs draw their note sequences from the same MAESTRO scores, so both overlap TransKun's training data. `input_type: audio` runs go further and feed TransKun the actual MAESTRO recordings it trained on, not just the same notes. hFT-Transformer is also trained on MAESTRO V3, so both cases apply to it too.

Note on `e-gmd`: [datasets.md](datasets.md) states that Sonitra's transcription and scoring target pitched instruments, not drum hits, so E-GMD is not benchmarkable by any of the three models. This is a tooling limit, not a difference between them.

Neither piano model refuses non-piano input. Point either at an orchestral or guitar recording and it returns plausible-looking, fully scoreable MIDI, complete with pitches, onsets, offsets and velocities. Nothing signals that the input was out of scope. A silent category error like this is worse than a crash, because a crash cannot be mistaken for a result.

### Basic Pitch

**Task.** Polyphonic note transcription and pitch estimation for many instruments. Polyphonic means many notes at once. The model outputs note events with pitch, start and end time, loudness and pitch bends. It works best on one instrument at a time.

**Architecture.** Lightweight convolutional neural network with harmonic stacking. The input is a constant Q transform. A constant Q transform is a pitch spaced spectrogram. The network predicts three heads at once: onsets, frames and contours. From those it builds notes and pitch bends. It is not a Transformer.

**Training data.** The original work trains on a mix of curated note level sets that include many instruments and singing. The exact split of training and held out sets for the released weight has changed since the paper. This page does not list a fixed count.

**Publication.** Bittner et al., ICASSP 2022 (Bittner, Bosch, Rubinstein, Meseguer-Brocal and Ewert, ICASSP 2022). The authors note the released code has been improved since the paper, so cite the code when you use its output.

**Licence.** Apache-2.0. Copyright 2022 Spotify AB. See the repository licence and the notice that the model data is subject to the same terms as the code distribution.

**Output fidelity.** Velocity is `round(float(amplitude) * 127)` (`src/sonitra/transcribe/basic_pitch.py`), where `amplitude` is the model's own note-activation confidence, not a measured loudness. Treat it as a rescaled confidence score, not a calibrated MIDI velocity. The model can also compute pitch bends, but `src/sonitra/midi_writer.py` writes no pitch-wheel MIDI messages, so bends never reach the output file. Setting `multiple_pitch_bends: true` only changes which overlapping same-pitch notes the model is allowed to emit; it does not add glissando curves, continuous slides between pitches, to the written MIDI.

**Numeric reproducibility.** Numbers from this card are comparable only within one device and one numeric mode. On `basic_pitch`, CPU and GPU disagreed on one note event in 1781 on identical input (1781 events on CPU, 1780 on GPU), and changing the GPU batch size shifted the model's note activations by up to 1.29e-02, against 8.05e-07 for the same change on a CPU. An activation here is a probability; `model_output_to_notes` thresholds onsets at 0.5, so a shift that size flips notes and changes F1. `src/sonitra/transcribe/basic_pitch.py` applies the remedy, `transcription.numeric_mode` (`off`, `warn` or `strict`; default `off`), and `src/sonitra/config.py` declares it. Record the device and mode with any published number. See [reproducibility.md](reproducibility.md).

| Property | Value |
|---|---|
| Sample rate | Any input rate. Resampled to 22.05 kHz before inference |
| Channels | Down mixed to mono at predict time |
| Model format | TensorFlow, also shipped as CoreML, TFLite and ONNX |
| Default model path | `ICASSP_2022_MODEL_PATH` from the `basic-pitch` package |
| Input length | Any length. The model windows internally |
| Pitch range | MIDI 21 to 108 with pitch bend support |
| Package | `basic-pitch` 0.4.x. Installed by default with Sonitra |
| Device config | Unified `cpu`, `cuda`, `cuda:N`, `GPU:N`; `cuda` is translated to TensorFlow's `GPU:0` internally |

### TransKun

**Task.** Piano only transcription. The model turns a piano recording into MIDI notes with pitch, start, end and velocity. Velocity is loudness from 1 to 127.

**Architecture.** Transformer encoder plus a neural semi conditional random field. The encoder scores every possible time interval for being an event. The semi CRF layer decodes those scores into a set of non overlapping note intervals with dynamic programming. V2 replaces the original score module with a scaled inner product and simplifies the noise score.

**Training data.** MAESTRO V3, a set of 1,276 piano performances with aligned MIDI and audio. The default checkpoint shipped in the wheel is trained with data augmentation and without pedal extended note lengths. Other checkpoints are listed on the GitHub model cards table, with scores on MAESTRO, MAPS and SMD. Pedal extension means stretching each note to its sustain pedal release. The shipped weight skips that step.

**Publication.** Yan and Duan, ISMIR 2024 (Yan and Duan, ISMIR 2024) for V2. The V1 framework is Yan, Cwitkowitz and Duan, NeurIPS 2021 (Yan et al., NeurIPS 2021).

**Licence.** MIT. Wheel `transkun==2.0.1` bundles a 56 MB checkpoint at `transkun/pretrained/2.0.pt` with its conf at `pretrained/2.0.conf`. No download at runtime.

**Output fidelity.** TransKun predicts real velocity, trained against MAESTRO's recorded key-strike velocities, so its numbers mean something different from Basic Pitch's rescaled confidence score (see Velocity above); a side-by-side velocity comparison between the two is not meaningful. The checkpoint bundled in the wheel is trained without pedal-extended note lengths (see Training data above), so its note ends are literal key releases, not sustain-pedal releases. Reference MIDI that holds the sustain pedal down will disagree with TransKun systematically on note duration, which shows up in the onset+offset and key-overlap-ratio metrics (see [evaluation.md](evaluation.md)).

**Numeric reproducibility.** Numbers from this card are comparable only within one device and one numeric mode. On a 20-file `bsed` subset (`--limit 20`, `batch_size=1`), CPU and GPU disagreed on 4 files, the largest gap being frame F1 0.0069, with smaller differences on note onset F1, onset+offset F1 and velocity (all at most 0.0057). This generalises the `basic_pitch` finding above to a Transformer plus semi-CRF backend: the semi-CRF decodes interval scores by dynamic programming, a Viterbi-style search that amplifies small score perturbations differently from a per-frame threshold. The gap is about 115 times smaller than the largest real condition effect on those same keys (note onset F1 0.7922), so it is a comparability note, not a disqualifying effect. `src/sonitra/transcribe/transkun.py` applies the remedy, `transcription.numeric_mode` (`off`, `warn` or `strict`; default `off`), and `src/sonitra/config.py` declares it; with `strict` on both sides divergence was exactly 0, at negligible runtime cost (within 5%). Record the device and mode with any published number. See [reproducibility.md](reproducibility.md).

| Property | Value |
|---|---|
| Sample rate | 44.1 kHz. Loaded with `read_audio_resampled` via `pedalboard` `resampled_to(44100)` |
| Channels | Stereo or mono input. Passed as `(samples, channels)` to `model.transcribe` |
| Framework | PyTorch |
| Input length | Overlapping windows. Sonitra keeps the bundled conf as truth: 16 s window with 8 s hop when fields are `null`. CLI help shows 20 s and 10 s as its own defaults |
| Pitch range | MIDI 21 to 108. Raw output also contains -64 and -67 for sustain and soft pedal, filtered before `make_note` |
| Package | `transkun==2.0.1` under the `transkun` extra. `pip install 'sonitra[transkun]'` |
| Device config | `cpu`, `cuda`, `cuda:1`, `mps` or `GPU:0` style. Sonitra translates `GPU:0` to `cuda:0`. A `cuda` request with no CUDA raises |
| Checkpoint location | `importlib.resources` under `transkun/pretrained/`. Override with `weights_path` and `conf_path` |

### hFT-Transformer

**Task.** Piano only transcription. The model turns a piano recording into MIDI notes with pitch, start, end and velocity, the same four fields TransKun produces, and it is a direct point of comparison for TransKun: same instrument, same training corpus, same scoring.

**Architecture.** A two-level hierarchical frequency-time Transformer. The first hierarchy is a one-dimensional convolution along the time axis, then a Transformer encoder and a Transformer decoder along the frequency axis, where the decoder turns frequency bins into the piano's 88 pitches. The second hierarchy is another Transformer encoder, along the time axis. `n_stride` exists because a single pass over a window blurs notes near its edges: halving the stride at inference time and keeping only the central part of each pass is the paper's own remedy for that position-dependent accuracy fluctuation, and `0` disables it.

On top of the network the model predicts a piano-roll representation, which is a grid of activations marking each pitch at each moment, and decodes that grid into discrete note events. The decoder has two head sets, one per axis. `output: second`, the default, selects the time-axis set, which is what the model's paper reports; `output: first` selects the other.

**Training data.** MAESTRO V3, the same aligned piano performances TransKun trains on. The paper also evaluates on MAPS.

**Publication.** Toyama, Akama, Ikemiya, Takida, Liao and Mitsufuji, ISMIR 2023 (Toyama et al., ISMIR 2023), "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer". Upstream is research code at <https://github.com/sony/hFT-Transformer>.

**Provenance.** The network and the note decoding are vendored under `src/sonitra/transcribe/_hft/`, with upstream's `LICENSE` alongside, pinned at commit `71a2ee06e9ced1ea24673c95ee0acded2fc98d04`. Vendoring is necessary rather than decorative: upstream has no package metadata, so it cannot be installed as a dependency, and its top-level module name would collide with another package. Sonitra therefore cannot drift from the pinned code without a deliberate change to the vendored copy.

**Licence.** The vendored code is MIT, Copyright 2023 Sony Group Corporation. The release asset that carries the weights states no licence of its own, and the weights are derived from MAESTRO, which is CC BY-NC-SA 4.0. That is why Sonitra neither redistributes the weights nor commits them: users fetch them themselves, and the non-commercial share-alike terms of the training data travel with them. The same MAESTRO licence is named in the README's third-party section.

**Installation.** Torch is required, which the existing `transkun` extra already provides, so there is no new extra and no dependency change. The weights are a separate one-time step, because the released checkpoint ships as a pickled `nn.Module` whose tensors are tagged `cuda:0` and therefore cannot be loaded at runtime. The setup script downloads the pinned release asset, verifies its size and sha256 against the registry, converts it once into a `model.pt` that `torch.load(weights_only=True)` accepts, and writes a `manifest.json` recording the provenance next to it:

```bash
python scripts/setup_hft_transformer.py
```

Run it once. The devcontainer and the production image run it for you on start (see [docker.md](docker.md) and [devcontainer.md](devcontainer.md)). The script is idempotent: a second run reports the existing install and exits. `--models-dir PATH` overrides the target directory and `--archive PATH` installs from an archive you already downloaded. Its own `--help` documents the exit codes: `0` installed or already installed, `1` failure, `2` usage error, `3` torch is not installed. Torch is checked before any download starts, so an image built with `INSTALL_TRANSKUN=0` skips the step quietly rather than reaching for the network.

**Output fidelity.** Velocity is a predicted value in 1 to 127, trained against MAESTRO's recorded key-strike velocities, so it means the same thing as TransKun's and not the same thing as Basic Pitch's (see Velocity above). Notes whose decoded offset lands past the end of the audio are kept rather than clipped, because the decoder pads the window tail so the final note can terminate and truncating it would disagree with the released model. The count is recorded as `offset_past_end` in metadata so the effect is visible. Every transcribe records `output_head`, `n_stride`, `batch_size`, `thresholds`, `resampler: pedalboard`, `sample_rate: 16000`, `upstream_commit`, `checkpoint`, `weights_sha256`, `note_events_total`, `notes_kept`, `filtered_dropped` and `offset_past_end`, alongside the device and numeric-mode keys the other backends record.

**Numeric reproducibility.** Numbers from this card are comparable only within one device and one numeric mode. Two results bound the range. On CPU at `batch_size=1` the vendored path is bit-for-bit identical to the upstream reference implementation, across 43 windows and both window modes, so the port itself introduces no numeric change. CPU against CUDA diverges: up to 3.3e-04 on the velocity logits, which changes 70–75 of roughly 75 note-list entries in the last bits. As the Basic Pitch and TransKun cards say, record the device and the `transcription.numeric_mode` with any published number, and see [reproducibility.md](reproducibility.md). One more measurement bears on the `batch_size` default: CPU `batch_size=4` against `batch_size=1` was bitwise identical on that machine, so batching is not recommended here for the memory reason in caveat 3 below, not because it changes the numbers.

One difference from upstream is deliberate. Sonitra resamples with `pedalboard` rather than torchaudio, because upstream's audio load now depends on a torchaudio codec backend that is not installed. That changes the numbers: log-mel values move by as much as 5.4. It does not degrade the result. Onset F1 came out 0.0019–0.0028 higher through Sonitra's path than through upstream's. Benign as measured, but the audio path is not numerically the upstream one, and a paper's published figure is not directly comparable to a Sonitra run.

| Property | Value |
|---|---|
| Sample rate | 16 kHz, pinned by the feature extractor. Resampled with `pedalboard` `resampled_to(16000)` via `read_audio_resampled` |
| Channels | Stereo or mono input. Mean down to mono at the pinned feature rate |
| Framework | PyTorch, plus torchaudio for the log-mel transform (2048-point FFT, 256 mel bins, 256-sample hop) |
| Input length | Overlapping windows. `n_stride` 0 to 64 in frames of 16 ms; `0` is a single pass over the whole window |
| Pitch range | MIDI 21 to 108 |
| Decoder head | `output: second` (time-axis, the default and the paper's) or `output: first` (frequency-axis) |
| Thresholds | `onset_threshold`, `offset_threshold`, `mpe_threshold`, each default 0.5, in (0, 1] |
| Batch | `batch_size` default 1. See the throughput note below before raising it |
| Package | Under the existing `transkun` extra, so `pip install 'sonitra[transkun]'` is all you need. No new extra |
| Device config | Unified `cpu`, `cuda`, `cuda:N`, `GPU:N`; `GPU:N` is translated to `cuda:N` |
| Checkpoint location | `<models_dir>/hft_transformer/maestro/model.pt` plus a `manifest.json`. `weights_path` overrides it and must point at a converted file, never the upstream `.pkl` |

**Measured performance.** Measured on the maintainer's machine (RTX 4090 Laptop GPU, CPU inference via PyTorch 2.12.1) against three MAESTRO v3 test-split pieces rendered to audio from MIDI with a GM SoundFont. **These are rendered-audio numbers, not real-recording numbers.** The container used for the measurement has MIDI only, so the comparison against the real MAESTRO recordings is outstanding and has not been run.

| metric | hFT-Transformer | TransKun |
|---|---|---|
| note onset F1 (duration-weighted) | **0.9746** (per file 0.9730–0.9801) | 0.9965 |
| note onset+offset+velocity F1 (duration-weighted) | 0.673 | 0.796 |
| note onset+offset F1 | 0.906 | 0.944 |

Three caveats travel with these numbers, and each one matters more than the table does.

1. **Velocity is the weak axis.** 0.673 against TransKun's 0.796. Onset detection is close to parity, and velocity is not. Anyone choosing between these two models on this evidence should read the velocity gap, not the onset gap.
2. **`note.onset_offset_f1` for this model is a lower bound, and the reason is not the model.** Its training references are pedal-extended: the reference builder forces note-off at the CC64 sustain-pedal release, so a note rings until the pedal lifts. Sonitra's MIDI reader is pedal-blind — it has no `control_change` branch — so the reference durations here are 22.2 % shorter than the model's own training references (+993.5 s of reference note length measured against +180.4 s). Onset F1 is the convention-neutral number for this comparison; onset+offset is depressed by a reference-construction difference, not only by model quality. Do not read 0.906 against 0.944 as a clean capability comparison.
3. **Throughput is fine and batching is not worth it.** CPU real-time factor 0.43–0.44, roughly 2.3× faster than realtime, and the pure-Python note decoder is only 1.9–2.1 % of CPU time, so no throughput work is needed. `batch_size` therefore stays at 1 by default: batching buys almost nothing on CPU while multiplying peak resident memory about 4.2×.

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

These are baseline-condition results only, from benchmark runs on the maintainer's machines. Scores depend on the corpus, the input mode and the render settings, not on the model alone, so no single number here should be quoted out of that context. They also depend on the device and on `transcription.numeric_mode`. As [docker.md](docker.md) says of its two R versions, do not mix results from different devices or modes in one study. See [reproducibility.md](reproducibility.md) for the measured size of those differences.

The `corpus/` directory is not tracked in git, so you cannot regenerate this table from a fresh checkout. Each run's own `summary.json` records its host and timing, for anyone who needs that detail.

TransKun is absent from the table when it has no qualifying benchmark run yet. That is not a comment on how it performed; the generated block below names it explicitly when this is the case.

The `Selection` column records which files each run scored. `all` means the full corpus, with no filter. A label such as `split=test` means the run kept one filtered subset of the corpus, chosen using the dataset's metadata. The table leaves out runs that used a random sample, because a random subset is not a baseline. For a model trained on a corpus's train split, its rows for the held-out splits (splits the model did not train on) are the fair comparison, because the `all` row includes files that model trained on.

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

| Corpus | Input | Selection | Model | n | onset F1 | +offset F1 | +vel F1 | vel corr | frame F1 |
|---|---|---|---|---|---|---|---|---|---|
| bsed | audio | all | basic_pitch | 100 | 0.056 | 0.016 | 0.011 | -0.092 | 0.300 |
| bsed | midi | all | basic_pitch | 20 | 0.641 | 0.333 | 0.206 | -0.107 | 0.676 |
| guitarset | audio | all | basic_pitch | 720 | 0.772 | 0.546 | 0.546 | — | 0.827 |
| guitarset | midi | all | basic_pitch | 360 | 0.860 | 0.298 | 0.298 | — | 0.787 |
| maestro-v3 | audio | all | basic_pitch | 1276 | 0.662 | 0.109 | 0.050 | 0.031 | 0.535 |
| maestro-v3 | midi | all | basic_pitch | 1276 | 0.817 | 0.136 | 0.059 | 0.055 | 0.661 |

_`transkun` is omitted: no qualifying benchmark run. Its only baseline data is `maestro-v3/transkun_baseline` (n=2), `maestro-v3/transkun_baseline_batch16_strict` (n=2), `maestro-v3/transkun_baseline_gpu` (n=2), `maestro-v3/transkun_baseline_gpu_batch8_strict` (n=2), `test/transkun_baseline` (n=2), too little to compare fairly._

_Baseline condition only, from 6 run(s) under `corpus/`; newest run 2026-09-12. Each run's own `summary.json` carries its host and timing._
<!-- END GENERATED: baselines -->

---
[← Back to README](../README.md)
