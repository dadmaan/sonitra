# Roadmap

This file lists work Sonitra has not built yet. What is built is recorded in
`CHANGELOG.md`, and how the tool behaves today is described in `docs/`.

## More instrument sets

**Status:** Not started.

`scripts/download_datasets.py` registers 13 keys today: MAESTRO V3, BSED,
MusicNet, E-GMD, GuitarSet and GAPS. Slakh2100 and other multi-instrument sets
are not among them. Why it matters: every set except E-GMD is pitched, so the
one drum set the script can fetch is download-only, because the transcription
and evaluation backends score pitched instruments. What blocks it: a new key
needs its own source list, its target folders and, where the ground truth is not
MIDI, its own converter, the way each current set has one.

## GuitarSet hex-pickup stems

**Status:** Deferred, not started.

GuitarSet ships 6-channel hex-pickup tracks alongside its two single-channel
variants. All three GuitarSet keys say the stems are deferred in their own help
text, and `docs/datasets.md` repeats it. Why it matters: the per-string pickup
carries labels down to string and fret, so it is the only source in the tree
for per-string guitar ground truth. What blocks it: the fetch is one source
entry, but nothing downstream decides what a 6-channel file should become.
`read_audio` returns every channel as found, and `write_audio` writes back
whatever channel count it is handed, so mixing down or selecting a channel is an
open choice, not an oversight.

## GAPS with `sonitra evaluate` and the batch script

**Status:** Deferred, nothing in the data blocks it.

GAPS runs with `sonitra benchmark` today. Its stems match across folders, so
`sonitra evaluate` and `scripts/run_transcribe_eval.py` would work on it as
well; `docs/datasets.md` says the same and explains why GuitarSet cannot. Why
it matters: GAPS is the longest real-recording set in the tree, about 23 h of
classical guitar from studio to phone microphone, with score MIDI already lined
up to the audio. What is left: `scripts/run_transcribe_eval.py` runs the presets
under `config/examples/`, so the wiring is a preset for GAPS or a dataset
argument passed to the script.

## BSED note-level alignments

**Status:** Not started, and no longer blocked on anything upstream.

BSED ships hand-checked note alignments between each score and its real
recordings, and the downloader leaves them on the server: the `bsed` entry's
`extract_map` has two targets, the score MIDI and the concert recordings, and
its help text says the alignment files are not fetched. Why it matters: the
alignments are a truer scoring target than raw score MIDI, because real players
drift from what is written. What it needs: an `annotations/` target in that
entry, pointing at the note-alignment folder in the archive, which is the same
corpus folder the GuitarSet converter already writes; and a reader under
`evaluation/` for the alignment format, alongside the MIDI reader. Scoring them
means comparing against a recording rather than a score, which
`render_pipeline.input_type: audio` now allows.

## Additional transcription backends

**Status:** Planned.

Five types are registered in `src/sonitra/transcribe/configs.py`: `basic_pitch`,
`transkun`, `hft_transformer`, `precomputed` and `external_command`. MT3 and
Omnizart are the intended next ones and appear nowhere in `src/`.

## Numeric reproducibility: the default and the published rows

**Status:** Two open items.

`transcription.numeric_mode` defaults to `off`. What the setting does and what
was measured is in `docs/reproducibility.md`; the numbers are not repeated here.

- Reconsider whether `off` should stay the default. Deterministic runs cost
  time, so the question is whether published work should pay that cost by
  default, and the answer should be a decision rather than a leftover.
- Carry the device and the numeric mode on each published baseline row. A
  summary row holds the condition, the transcriber, the counts, the overrides
  and the metric means, and nothing else, and the generated baseline table in
  `docs/model-cards.md` has no column for either. Both keys are already recorded
  per row in `benchmark_results.jsonl`, so the data exists and the tables do not
  show it.

## Parallel FluidSynth rendering

**Status:** Open, and the only one left in this area.

Rendering inside a condition runs in a thread pool only when the synth backend
is `pedalboard_instrument`; every other backend makes sound one file at a time,
and DawDreamer is forced to one worker because JUCE global state is not
thread-safe. Why it matters: a multi-condition run spends most of its wall time
rendering, so `benchmark.max_workers` currently scales transcription and leaves
the synthesis in front of it serial. What blocks it: FluidSynth is an external
command invoked once per file, so parallelism means either several processes
each running their own FluidSynth, or a FluidSynth that renders a batch in one
call. Nothing on the transcription side blocks it any more: GPU memory growth is
a setting that is applied before the model loads, and a pool worker now builds
each transcriber once for the lifetime of that worker.

## Vintage recording chains: additive noise and wow and flutter

**Status:** Not started.

`config/benchmark/old_recording/vintage_scenarios.yaml` models three old
recording chains with filters, tone colour, saturation and level control, and
its README lists what the preset leaves out. Why it matters: surface noise,
hiss and mains hum are plausibly a larger driver of transcription failure than
anything the preset does model, so it cannot answer the question a reader will
ask of it. Open questions:

- How does the pipeline express a stage that is not a native plugin? The effects
  chain groups native plugins into `pedalboard.Pedalboard` segments, and the one
  stage that works on plain arrays, the tuning offset, gets a segment of its own.
  Noise, crackle and hum would follow that shape, which means a new config
  section rather than a new plugin.
- Where do the parameter values come from? Crackle rate and strength, the AM
  noise floor and hum level, and the tape wobble rates have no measured source
  yet. Guessed numbers would not survive review, and the filter cutoffs in the
  current preset did not either; they were solved numerically against the
  installed plugin.
- How is wow and flutter tuned? It is a variable-rate resample rather than a
  stationary filter, so it needs its own stage, and its calibration target is
  DIN 45507, which the README already points at.
- Where do the two new stages sit relative to the chain? Speed change has to run
  before it and the noise after it, so the ordering is part of the design.

## Documentation site

**Status:** Settings written, no site built.

`zensical.toml` carries the site settings and two palettes, and its own comment
records that navigation is not configured yet. `docs/` holds 17 pages. What is
missing, in the order it would have to happen:

- `index.md`, adapted from `README.md`, as the home page.
- A `nav` block in `zensical.toml`, once the page set is final.
- Zensical added to the project's dependencies. It is not in `pyproject.toml`
  today, and nothing in the tree builds a site.
- A build check, so a relative link that no longer resolves fails loudly instead
  of shipping a dead page.
- A decision on where the site is hosted.

Planned structure, with every existing page placed exactly once:

| Group | Pages |
|---|---|
| Home | `index.md` (new, from `README.md`) |
| Getting started | `devcontainer.md`, `docker.md`, `cli.md`, `configuration.md` |
| Datasets | `datasets.md`, `custom-datasets.md` |
| Benchmarking and evaluation | `evaluation.md`, `statistical-analysis.md`, `reproducibility.md`, `model-cards.md` |
| Extending Sonitra | `adding-a-transcriber.md`, `plugins.md`, `python-api.md`, `rest-api.md` |
| Research notes | `abstract.md`, `research.md`, `gpu-batching-basic-pitch.md` |
