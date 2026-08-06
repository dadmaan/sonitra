# Configuration

`config/source.yaml` in the repository is the fully-annotated reference config documenting every available parameter. Copy it as a starting point, or generate a minimal starter config with:

```bash
sonitra init --config config.yaml
```

The config file is validated by Pydantic — unknown keys are hard errors.

Key sections and their purpose:

| Section | Controls |
|---|---|
| `pipeline` | Synth backend (`synth_backend`), effects chain (`effects_chain`), BPM, sample rate, bit depth, channels, parallelism (`max_workers`) |
| `io` | `corpus_root` (base path), `dataset` (scopes all paths under `corpus_root/{dataset}/`), output format (`wav`, `flac`, `mp3`), file naming template |
| `dawdreamer` | Faust script path, VST3 plugin path, preset path — required when `synth_backend: dawdreamer_vst`; `plugin_path` must NOT be set for `synth_backend: dawdreamer_faust` |
| `fluidsynth` | `soundfont_path` — path to the `.sf2` SoundFont file; required when `synth_backend: fluidsynth` |
| `pedalboard` | Pedalboard effects chain (`pedalboard.effects`); `pedalboard.instrument` sub-section configures the VST3 instrument plugin for the `pedalboard_instrument` backend |
| `normalisation` | Peak or RMS normalisation, target dB, pre/post effects |
| `quality_gates` | Silence, clipping, and minimum duration checks |
| `transcription` | Transcriber backends, output directory, per-backend thresholds |
| `evaluation` | Metric families to enable, tolerance values |
| `benchmark` | Conditions, parameter sweeps, baseline name |
| `observability` | JSONL manifest, failed-file list, SSE event emission |

## Synthesis backend and effects chain

**`pipeline.synth_backend`**

| Value | Synth engine | Requires |
|---|---|---|
| `dawdreamer_faust` | DawDreamer + built-in Faust oscillator | — |
| `dawdreamer_vst` | DawDreamer + VST3 instrument | `dawdreamer.plugin_path` |
| `fluidsynth` | FluidSynth CLI + SoundFont | `fluidsynth.soundfont_path` |
| `pedalboard_instrument` | Pedalboard VST3 instrument | `pedalboard.instrument.plugin_path` |

**`pipeline.effects_chain`**

| Value | Behaviour |
|---|---|
| `none` | No effects processing after synthesis |
| `pedalboard` | Apply the `pedalboard.effects` chain after synthesis |

## Transcription backends

| Backend | `type` value | Notes |
|---|---|---|
| Spotify Basic Pitch | `basic_pitch` | Installed by default; supports `device` field (default: `cpu`; set to `GPU:0` for GPU inference — requires `[gpu]` extras on Linux x86_64) |
| Pre-exported MIDI | `precomputed` | Point at a directory of MIDI from external tools |
| Any CLI tool | `external_command` | Template: `"tool transcribe {input} -o {output}"` |

## Built-in audio effects

Compressor, Reverb, Limiter, Chorus, Delay, Distortion, Gain, VST3 plugin. Each is a named entry under `pedalboard.effects` with an `enabled` flag. VST3 plugins are loaded at their factory default settings; parameters cannot be set from YAML.

---
[← Back to README](../README.md)
