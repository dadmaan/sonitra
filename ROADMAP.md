# Roadmap

This file lists planned work Sonitra has not yet built.

## Stem separation

**Status:** Not yet built.

Sonitra plans Demucs-based stem separation as an optional step between sound creation and transcription. Demucs is a tool that splits mixed audio into separate tracks. A stem is one isolated instrument track. When turned on, it would pull out the target instrument before transcription. This helps test AMT systems on realistic mixed music. AMT means automatic music transcription, turning audio into notes.

Planned design:
- `SeparatorProtocol` with a `separate(audio, sample_rate) -> audio` interface
- `@register_separator("demucs")` decorator following the existing pluggable-backend pattern
- `make_separator(cfg)` factory with lazy import of `demucs` so the optional dependency is never loaded at package import time
- `separation` section in `PipelineConfig` controlling model name, device (`cpu`/`cuda`), and enable flag
- `demucs` optional extra in `pyproject.toml` (already declared, not yet wired to a backend)
- Docker profile `with-demucs` for the image variant that includes the extra (~1 GB overhead)

The pipeline picture will change to:
```
MIDI → audio synthesis → stem separation → transcription → evaluation vs. reference
```
MIDI means a digital score file.

## Additional datasets and instruments

**Status:** Piano, orchestral, chamber/orchestral, drum, and guitar sets
(both short clips and long-form) are ready. More sets are not yet started.

`scripts/download_datasets.py` now fetches MAESTRO V3.0.0 (piano, `-midi`,
`-wav`, and `-full` options), BSED (orchestral, Beethoven symphony clips, MIDI plus real
recordings), MusicNet (chamber and orchestral music with many instruments, `musicnet-midi` for score MIDI only, ~4 MB, or `musicnet-full` for 34 h of recordings plus label CSVs and a score-MIDI copy, ~10.6 GB, needs `scripts/musicnet_labels_to_midi.py` to build aligned MIDI), and the Expanded Groove MIDI Dataset (drums, `-midi` and `-full`; download-only,
because Sonitra tests pitched instruments like piano, not drum hits).
The script also has a picker table you can click through and parallel
downloads with `--jobs N`. Each entry can pull from more than one link or file type
(zip and tar.gz). It sorts files into `midi/`, `recordings/`, `metadata/`, or
`annotations/` (GuitarSet JAMS, a music-label format) by
file ending and name rules. The team looked at MAPS but dropped it. MAPS sits behind a
sign-up form with no direct download link, so the script cannot fetch it on its own. GuitarSet (acoustic guitar, `guitarset-mic` and
`guitarset-mix` single-channel options, or `guitarset-full` for both at once, JAMS
notes turned once into MIDI with
`scripts/guitarset_jams_to_midi.py`) is ready. Its 6-channel hex-pickup
tracks (`hex-pickup_original` and `_debleeded`) are on hold. `read_audio`
reads all channels and `write_audio` will not mix them down, so Sonitra needs render-path work first.
GAPS (Guitar-Aligned Performance Scores, `gaps-midi` for the MIDI alone or
`gaps-full` with the audio) is ready too: 404
classical-guitar recordings, about 23 hours, made everywhere from studios to
phone microphones, with score MIDI already lined up to the audio. It needs no
conversion step. `gaps-full` re-uses any MIDI `gaps-midi` already fetched. It is the first entry served from Hugging Face rather than
from one archive file, so the script gained a `hf_tree` source kind that lists
a folder through the Hugging Face tree API and fetches each file at a pinned
commit. Any future Hugging-Face-hosted set can reuse it.

GAPS works with `sonitra benchmark` today. It would also work with `sonitra
evaluate` and `scripts/run_transcribe_eval.py`, because its audio and MIDI
stems match exactly, unlike GuitarSet's suffixed stems. Wiring those two up
was left out of the first pass on purpose, not because anything blocks it.

Sonitra plans more instruments in future,
multi-instrument sets such as Slakh2100, and auto downloads for them.

## Additional transcription backends

**Status:** Not yet started.

Sonitra now uses Basic Pitch and a generic `external_command` tool. Basic Pitch is Spotify's free transcription tool. `external_command` lets you call any outside transcriber. Direct support for more AMT tools, such as MT3 and Omnizart, is being considered. MT3 is a Google multi-instrument transcription model.

## User-facing documentation site (Zensical)

**Status:** Not yet started.

Sonitra plans a docs site built with Zensical, a static site tool. The `zensical.toml` config file is ready. Planned pages:

- Home (`index.md`) is the landing page with quick-start links.
- Tutorial (`tutorial.md`) is a step-by-step walkthrough of a full benchmark run.
- Project structure (`project_structure.md`) describes the directory layout and key file roles.
- CLI reference (`cli_reference.md`) lists all commands, flags, and examples.
- API reference (`api_reference.md`) documents the Python API for library use.
- Design decisions (`design_decisions.md`) records architectural rationale and trade-offs.
- Troubleshooting (`troubleshooting.md`) lists common issues and solutions.

The nav part in `zensical.toml` stays commented out until the pages are written.

## Real-audio transcription mode

**Status:** Not yet started.

Sonitra now transcribes only audio it makes itself from MIDI. A planned mode
would send a dataset's own recordings straight to transcription, skipping
sound creation. This allows tests on real instrument audio, the standard AMT
test in past work. For MAESTRO V3.0.0 this means using the paired
`.flac` recordings instead of making new audio from MIDI. FLAC is a lossless audio format.

Planned design:
- Sonitra reads the dataset notes CSV, for example `maestro-v3.0.0.csv`, to link each score MIDI to its own `.flac` or `.wav` recording. CSV means comma-separated table. WAV is a common audio format.
- Sonitra skips sound creation. It scores transcription output against the score MIDI just as it does now.
- You need the full audio download (`maestro-v3-full` or `-wav` with `scripts/download_datasets.py`, about 120 GB). The MIDI-only set (`maestro-v3-midi`, about 57 MB) is enough for the make-audio-then-transcribe path.
- This differs from now, where sound quality itself is part of what you test.

BSED (see "Additional datasets and instruments" above) is an easier first target
than MAESTRO. `scripts/download_datasets.py bsed` already fetches real
recordings into `corpus/bsed/recordings/` next to `corpus/bsed/midi/`. Both sides
share a `BSED-<NN>_...` name start. So Sonitra can link a score to its recordings by
file name, with no CSV lookup.

## Note-level performance-alignment annotations (BSED)

**Status:** Not yet started.

BSED ships hand-checked note alignments between each score and its real
recordings under `03_NoteAnnotations/Note-Level-Alignment/` (CSV plus `.npz`, a NumPy data file). It also has a
rougher `Sequence-Alignment/` set. `scripts/download_datasets.py` fetches neither today. Each file links a score note to its start and end time in one
real recording. This is a truer answer than the plain MIDI-score timing
Sonitra uses today, because real players drift from the score
(rubato, which means stretching time, systematic delays, and more).

Planned use: once real-audio mode (above) is ready, scoring could
use the per-recording aligned file as truth instead of raw
score MIDI. This gives a tighter measure for orchestral transcription. This
needs two code changes. Extend the `bsed` entry's `extract_map` in
`scripts/download_datasets.py` with an `annotations/` target. And add a reader under `evaluation/` that reads the CSV and npz alignment format instead of
reading MIDI on its own.

## Parallel benchmark conditions on GPU

**Status:** Blocked on TensorFlow GPU memory setup. `benchmark.max_workers`
stays at 1 in every `config/benchmark/*.yaml` preset for now. TensorFlow is the math library Basic Pitch uses. `max_workers` sets how many tasks run at once.

`benchmark.max_workers` is the only setting that runs `sonitra benchmark` tasks at once. It
uses a `ProcessPoolExecutor` over test setups (see `benchmark/runner.py`), not a thread
pool. Each worker is thus a separate process with its own TensorFlow copy.
Nothing in `src/` calls `tf.config.experimental.set_memory_growth`, so TensorFlow
keeps its default of taking almost all graphics memory at start. The
first process took about 22 GB of a 24 GB RTX 3090, leaving no room for a
second worker. Using `transcription.transcribers[].device: GPU:0` with
`benchmark.max_workers > 1` will likely run out of memory (OOM means out of memory). So you cannot use the two together
today.

`transcription.max_workers` and `evaluation.max_workers` still work. Both share threads in one process and one GPU session, and both are already above
1 in the presets. `sonitra transcribe` uses the first. `sonitra evaluate` uses the second. `sonitra benchmark` uses neither.

Planned design:
- Turn on memory growth before the model loads in `transcribe/basic_pitch.py`, so each
  process uses only what it needs (the ICASSP 2022 model is small). ICASSP is a speech and audio conference:
  ```python
  for gpu in tf.config.list_physical_devices("GPU"):
      tf.config.experimental.set_memory_growth(gpu, True)
  ```
  This must run before any GPU-using TF call in the process. It raises
  `RuntimeError` if the device is already set up. So it belongs at the top of
  the slow-import block, with a guard, not at module load time.
- Sonitra may make this a setting instead of always-on behavior, for example a
  `gpu_memory_growth` flag on the transcription part. Memory growth trades
  some speed for the chance to share the card.
- Once done, raise `benchmark.max_workers` in the presets and measure. The gain
  depends on how much of each test's time is GPU transcription versus
  sound creation. Sound creation is one-at-a-time for the `fluidsynth` and `dawdreamer_*` tools these
  presets use (see `pipeline.max_workers` note below). FluidSynth turns MIDI into audio with a SoundFont. DawDreamer plays VST instruments.
- Sonitra may still cap parallel workers. N processes each loading a
  TensorFlow copy cost host RAM apart from GPU memory.

Related: Sonitra reads `pipeline.max_workers` only for the `pedalboard_instrument`
sound tool (see `pipeline.py`). Pedalboard hosts VST instruments here. `fluidsynth` and `dawdreamer_*` make sound one file at a time
no matter what. And `validate_worker_constraint()` resets the value to 1 for
DawDreamer because JUCE, the audio code behind it, cannot safely share threads. Parallel sound creation for the
FluidSynth command-line tool is a separate chance to improve.

## Vintage recording chains: additive noise and wow/flutter

**Status:** Not yet started. The full plan is at
`.local/notes/dev/20260810_exp_old_recording_config/PLAN_PHASE2.md`. It needs
approval and several open choices before build.

`config/benchmark/old_recording/vintage_scenarios.yaml` now models three
old recording chains (78rpm shellac discs, early reel-to-reel tape, AM radio
broadcast) as a steady, filter-based sound path.
It uses text-book filters, tone color, saturation, and
loudness control, all built from `pedalboard` plugins. Pedalboard is the audio-effects library. It leaves out
added noise (surface hiss, tape hiss, mains hum) and wow and flutter (wobble from
uneven tape or disc speed). Both likely hurt AMT transcription more
than anything the current config tests. AMT means turning audio into notes. See that config's README for the
full story.

`pedalboard.Plugin` is a fast native class tied to Python with pybind11, and `pedalboard.Pedalboard()`
is a native list that holds one type. So Sonitra cannot add noise or changing
effects as new `EffectConfig` or `chain_builder` options. They need
a separate path based on plain numpy and scipy arrays. NumPy and SciPy are Python math tools.

Planned design:
- New `src/sonitra/effects/vintage_effects.py` (settings classes: `NoiseFloorConfig`,
  `CrackleConfig`, `HumConfig`, `WowFlutterConfig`, kept apart from `EffectConfig`) and `vintage_chain.py` (`apply_vintage_effects`
  works over plain numpy arrays).
- New `VintageEffectsSection` on `PipelineConfig` with `pre_chain` and `post_chain`
  lists and a `seed` field. You can reach it with the current dotted-path benchmark
  change system with no runner code change.
- Two new hook points in `pipeline.py`'s shared `_render_file`. Wow and flutter
  (speed-change resampling) run before the pedalboard effects chain. Hiss,
  crackle, and hum run after it and before loudness fix-up. Both lists start
  empty, so all current configs sound the same.
- Fixed random starts per file (SHA256 hash of `f"{seed}:{midi_path.stem}"` feeds
  `np.random.default_rng`) so reruns match and `benchmark.resume` still works. SHA256 is a hash that turns text into a fixed code.
- A new partner scenario file, `vintage_scenarios_full.yaml` (7 setups:
  baseline plus 3 eras times 2 strengths, reusing the current chains plus
  the new noise and wobble effects). The current file's 7-setup deal
  stays the same.
- Wow and flutter tuning against DIN 45507 (a weighted loudness test for speed wobble, not a
  raw wave depth) is its own task. It mirrors the
  filter tuning already done for the current tone-limiting config.

Several settings (crackle rate and strength, AM radio noise floor and hum
level, tape wobble rates) have no source yet. The plan doc marks them as open
research instead of giving guess values. A later
`REFERENCE_PHASE2.md` source check would settle these before Sonitra locks the
final numbers.
