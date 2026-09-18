# Model cards

Sonitra benchmarks audio to note transcription. This page describes the two real model backends you can run. A model backend is the tool that turns audio into notes. A checkpoint is the saved weights for a trained model.

Both models take audio and output MIDI notes. MIDI is a digital score format with pitch, start time, end time and loudness. The page lists what each model was built for, how it works, what data it saw, where it was published, and its licence. The table at the end of each section gives the compact specs you need to run it.

MAESTRO overlap note: TransKun was trained on MAESTRO V3 and Sonitra ships a `maestro-v3` corpus. Sonitra renders and degrades its own audio from the MIDI scores so the model never hears MAESTRO recordings during a benchmark, but the note sequences still overlap because both come from the same scores.

Input resolution note: Basic Pitch resamples all input to 22.05 kHz. TransKun requires 44.1 kHz via `read_audio_resampled`. A side by side run therefore does not control for input resolution. A gap between the two may reflect the resampling as well as the models.

### Basic Pitch

**Task.** Polyphonic note transcription and pitch estimation for many instruments. Polyphonic means many notes at once. The model outputs note events with pitch, start and end time, loudness and pitch bends. It works best on one instrument at a time.

**Architecture.** Lightweight convolutional neural network with harmonic stacking. The input is a constant Q transform. A constant Q transform is a pitch spaced spectrogram. The network predicts three heads at once: onsets, frames and contours. From those it builds notes and pitch bends. It is not a Transformer.

**Training data.** The original work trains on a mix of curated note level sets that include many instruments and singing. The exact split of training and held out sets for the released weight has changed since the paper. This page does not list a fixed count.

**Publication.** Bittner et al., ICASSP 2022 (Bittner, Bosch, Rubinstein, Meseguer-Brocal and Ewert, ICASSP 2022). The authors note the released code has been improved since the paper, so cite the code when you use its output.

**Licence.** Apache-2.0. Copyright 2022 Spotify AB. See the repository licence and the notice that the model data is subject to the same terms as the code distribution.

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

---
[← Back to README](../README.md)
