# Plugins and SoundFonts

## VST3 plugin (optional)

Sonitra supports any VST3 instrument plugin for synthesis. [Vital](https://vital.audio/) is the tested, recommended free option.

1. Download Vital from [vital.audio](https://vital.audio/) and extract the archive.
2. Place the extracted plugin folder under `plugin/`:

```
plugin/
  vital/
    lib/
      vst3/
        Vital.vst3
```

3. Set `dawdreamer.plugin_path` and `render_pipeline.synth_backend` in your config:

```yaml
render_pipeline:
  synth_backend: dawdreamer_vst   # required when using a VST3 instrument plugin

dawdreamer:
  plugin_path: plugin/vital/lib/vst3/Vital.vst3
```

## Presets (optional)

VST3 preset files (e.g. `.vital` files for Vital) go under `preset/`:

```
preset/
  vital/
    MyPreset.vital
```

Set `dawdreamer.preset_path` in your config:

```yaml
dawdreamer:
  preset_path: preset/vital/MyPreset.vital
```

## SoundFont fallback (optional)

For SoundFont-based synthesis without a VST3 plugin:

```bash
# Linux
sudo apt install fluid-soundfont-gm

# macOS
brew install fluid-synth
```

Then set `render_pipeline.synth_backend: fluidsynth` and `fluidsynth.soundfont_path: /usr/share/sounds/sf2/default-GM.sf2` (or the path on your system) in your config.

## Core dependencies installed automatically

| Package | Role |
|---|---|
| `dawdreamer` | Faust/VST audio synthesis engine |
| `pedalboard` | Audio effects and instrument plugin API |
| `basic-pitch >= 0.4, < 0.5` | Default AMT backend (Spotify Basic Pitch) |
| `mido` | MIDI file parsing |
| `fastapi` + `uvicorn` | REST API server |
| `pydantic` + `pyyaml` | Config validation and loading |
| `numpy` + `scipy` | Evaluation metric computation |

---
[← Back to README](../README.md)
