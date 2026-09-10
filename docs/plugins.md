# Plugins and SoundFonts

Sonitra turns MIDI scores into audio. MIDI is a digital score format that stores notes, timing, and loudness. You can pick how Sonitra makes the sound. Use a VST3 instrument plugin for plugin-based sound, or use a SoundFont for a simple free setup. VST3 is a common plugin format for virtual instruments and audio effects. A SoundFont is a file with sampled instrument sounds, ending in `.sf2`.

## VST3 plugin (optional)

Sonitra works with any VST3 instrument plugin for sound creation. [Vital](https://vital.audio/) is the free option Sonitra has tested, so start there if you are new.

1. Download Vital from [vital.audio](https://vital.audio/) and extract the archive.
2. Place the extracted plugin folder under `plugin/`:

```
plugin/
  vital/
    lib/
      vst3/
        Vital.vst3
```

3. Point your config to the plugin. Set `dawdreamer.plugin_path` and `render_pipeline.synth_backend` in your config. DawDreamer is the tool Sonitra uses to play the VST3 instrument:

```yaml
render_pipeline:
  synth_backend: dawdreamer_vst   # required when using a VST3 instrument plugin

dawdreamer:
  plugin_path: plugin/vital/lib/vst3/Vital.vst3
```

## Presets (optional)

A preset is a saved sound setting for your instrument. VST3 preset files (for example `.vital` files for Vital) go under `preset/`:

```
preset/
  vital/
    MyPreset.vital
```

Set `dawdreamer.preset_path` in your config to use it:

```yaml
dawdreamer:
  preset_path: preset/vital/MyPreset.vital
```

## SoundFont fallback (optional)

Use this path if you do not want to set up a VST3 plugin. FluidSynth is a free tool that turns MIDI into audio using a SoundFont:

```bash
# Linux
sudo apt install fluid-soundfont-gm

# macOS
brew install fluid-synth
```

Then set `render_pipeline.synth_backend: fluidsynth` in your config. Also set `fluidsynth.soundfont_path` to your SoundFont file, for example `/usr/share/sounds/sf2/default-GM.sf2`, or the path on your system.

## Core tools Sonitra installs for you

| Package | Role |
|---|---|
| `dawdreamer` | Plays sounds, using Faust (a built-in simple tone maker) or VST instruments |
| `pedalboard` | Adds audio effects and hosts instrument plugins |
| `basic-pitch >= 0.4, < 0.5` | Default transcription tool (Spotify Basic Pitch, turns audio back into notes) |
| `mido` | Reads and writes MIDI score files |
| `fastapi` + `uvicorn` | Runs the web server for the REST API |
| `pydantic` + `pyyaml` | Checks your config file for errors and loads it |
| `numpy` + `scipy` | Computes the evaluation scores |

---
[← Back to README](../README.md)
