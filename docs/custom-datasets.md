# Using your own dataset

You can point Sonitra at any folder of your own scores or recordings. This page covers the setup: folder layout, file naming, and the config settings that matter.

## What you need

Every piece needs a reference MIDI file: the score Sonitra marks your transcription against. Audio on its own is not enough, because Sonitra has nothing to compare the transcription to without a MIDI answer key.

## Folder layout

Pick a name for your dataset and create this layout under `corpus/`:

```
corpus/
  my-dataset/
    midi/         # required: one .mid or .midi file per piece
    recordings/   # optional: your own real audio, if you have it
    metadata/     # optional: a CSV with facts about each piece
```

Never put your recordings in `audio/`. That folder holds audio Sonitra renders itself, and Sonitra reads your recordings only from `recordings/`.

MIDI files end in `.mid` or `.midi`; audio files end in `.wav`, `.flac`, or `.mp3` (the ending match is not case-sensitive). Sonitra reads audio at any sample rate. It also searches subfolders at any depth inside `midi/` and `recordings/`, so you can organize pieces into subfolders if you want. If you keep your data outside `corpus/`, set `io.corpus_root` in your config to point there instead. You can also make `midi/` or `recordings/` itself a symbolic link to another folder; just don't symlink a folder *inside* them, since Sonitra skips those silently.

## If you have only MIDI files

This is the default mode, so an existing benchmark config such as `config/benchmark/benchmark_test.yaml` works as it is. Try it on two files first:

```bash
sonitra benchmark --config config/benchmark/benchmark_test.yaml --dataset my-dataset --limit 2
```

Sonitra renders audio from your MIDI, transcribes it, and scores the result against the same MIDI. Results go to `corpus/my-dataset/benchmark/<config-name>/`, here `benchmark_test/`. Drop `--limit` for the full run.

## If you have MIDI files and matching recordings

Copy a benchmark config, set these two lines in the copy, then run it the same way:

```yaml
render_pipeline:
  input_type: audio
evaluation:
  dtw:
    enabled: false
```

With `input_type: audio`, Sonitra applies each test condition's effects to your recordings instead of rendering new audio from the MIDI, so your SoundFont or synth settings no longer apply. `--limit 2` then picks two recordings. Turn off `evaluation.dtw`, since that check compares rendered audio against a re-render of the transcription and never runs in audio mode anyway; leaving it on just logs a warning. `config/benchmark/paper_experiments/guitar_only.yaml` is a working example of this mode.

## Naming recordings so they pair with their MIDI

Sonitra matches each recording to a MIDI file by name alone (subfolders don't count). It splits the name on underscores and compares from the start, case-sensitively. The safe rule: name a recording exactly like its MIDI file, optionally followed by `_` and any extra text. Also give every MIDI file a unique name, and make sure no MIDI name is another MIDI name plus `_` and more text, or the match becomes ambiguous.

| Recording | MIDI files present | Result |
|---|---|---|
| `song1_take1.wav`, `song1_take2.wav` | `song1.mid` | both pair to `song1.mid` |
| `piece_1.wav`, `piece_10.wav` | `piece_1.mid`, `piece_10.mid` | pair correctly |
| `song1-take1.wav` (hyphen) | `song1.mid` | no match |
| `Song1.wav` | `song1.mid` | no match (case differs) |
| `piece_1_live.wav` | `piece_1.mid`, `piece_1_arr.mid` | ambiguous, skipped |

A recording that finds no match, or more than one, is skipped with a warning in the log, and the run continues with the rest.

## Check your dataset before a run

Before a full run, check your files with `scripts/check_dataset.py`. It reads your dataset folder the same way `sonitra benchmark` does, using the benchmark's own pairing code, so its report matches what the benchmark will do. It changes nothing unless you tell it to.

```bash
python scripts/check_dataset.py --dataset my-dataset
```

Unlike the download script, this one needs the project environment, since it imports Sonitra. Add `--corpus-root DIR` if your data lives outside `./corpus`. It checks an audio-input run when `recordings/` holds audio, or a MIDI-input run otherwise; `--input-type midi` or `--input-type audio` overrides that.

The report groups results into errors, warnings, and notes, with up to 10 examples per group (`--verbose` lists them all). Errors cover unmatched or ambiguous recordings, clashing MIDI names, unreadable MIDI files, and missing MIDI or recordings. Warnings cover multiple instrument programs, drum-channel notes, files in `recordings/` with an ending Sonitra does not read, audio files left in `audio/`, and symlinked folders inside `midi/` or `recordings/`. Notes list MIDI files that no recording pairs with. The script exits with code 1 if it found any error, so you can run it before a long benchmark and stop early.

When a recording misses its MIDI file only by letter case or a separator such as a space, hyphen, or dot, the checker proposes a new name. For example, `Song1 - take1.wav` next to `song1.mid` becomes `song1_take1.wav`. It only proposes a rename when the new name matches exactly one MIDI file and no other file already has it; for a near miss such as the typo in `sogn1.wav` it only prints a "did you mean" hint. Renaming is two steps:

```bash
python scripts/check_dataset.py --dataset my-dataset --plan renames.csv
python scripts/check_dataset.py --dataset my-dataset --apply renames.csv
```

`--plan` writes the proposed renames to a CSV so you can read or edit them first. `--apply` checks every row before renaming anything, then renames the files and writes `renames.undo.csv`; run `--apply renames.undo.csv` to put the old names back. It never overwrites an existing plan or undo file, and it only ever renames recordings, never MIDI files, since the metadata join matches on MIDI names.

## Before you run

The checker above catches most of these for you, but it helps to know them anyway.

- If a MIDI file uses more than one instrument program, set `fluidsynth.program` (0 to 127) in your config. Otherwise the FluidSynth backend plays it with the SoundFont's default sound, usually piano.
- Remove drum tracks from your MIDI files. Sonitra's transcription and scoring target pitched instruments, so drum-channel notes get treated as ordinary pitched notes.
- Give every MIDI file a unique name, even across subfolders.
- Use `sonitra benchmark`, not `sonitra evaluate` or `scripts/run_transcribe_eval.py`. Those two match files by exact path and skip the name pairing described above.

## Adding facts about each piece

If you add a metadata CSV under `corpus/my-dataset/metadata/`, you can join it into a benchmark export the same way the built-in datasets do. See "[Joining dataset metadata into a benchmark export](datasets.md#joining-dataset-metadata-into-a-benchmark-export)" in [docs/datasets.md](datasets.md). The column you join on must hold the reference MIDI file's name.

---
[← Back to README](../README.md)
