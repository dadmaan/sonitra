# Roadmap

This file tracks planned features and capabilities that are not yet implemented.

## Stem separation

**Status:** Not yet implemented.

Demucs-based stem separation is planned as an optional preprocessing step between audio synthesis and transcription. When enabled, it would isolate the target instrument stem before feeding audio to the transcription backend — useful for benchmarking AMT performance under realistic polyphonic mix conditions.

Planned design:
- `SeparatorProtocol` with a `separate(audio, sample_rate) -> audio` interface
- `@register_separator("demucs")` decorator following the existing pluggable-backend pattern
- `make_separator(cfg)` factory with lazy import of `demucs` so the optional dependency is never loaded at package import time
- `separation` section in `PipelineConfig` controlling model name, device (`cpu`/`cuda`), and enable flag
- `demucs` optional extra in `pyproject.toml` (already declared, not yet wired to a backend)
- Docker profile `with-demucs` for the image variant that includes the extra (~1 GB overhead)

The pipeline diagram will update to:
```
MIDI → audio synthesis → stem separation → transcription → evaluation vs. reference
```

## Additional datasets and instruments

**Status:** Piano and orchestral corpora available; further expansion not yet started.

`scripts/download_datasets.py` currently supports MAESTRO V3.0.0 (piano, MIDI-only)
and BSED (orchestral — Beethoven symphony excerpts, MIDI + real recordings, see
below). Planned expansion to cover further instruments, multi-instrument datasets
(e.g. Slakh2100), and automated download support for further datasets.

## Additional transcription backends

**Status:** Not yet started.

Beyond Basic Pitch and the generic `external_command` backend, native integrations with other AMT systems (e.g. MT3, Omnizart) are under consideration.

## User-facing documentation site (Zensical)

**Status:** Not yet started.

Planned pages for a Zensical-generated documentation site (`zensical.toml` config exists):

- **Home (`index.md`)** — landing page with quick-start links
- **Tutorial (`tutorial.md`)** — step-by-step walkthrough of a full benchmark run
- **Project Structure (`project_structure.md`)** — directory layout and key file roles
- **CLI Reference (`cli_reference.md`)** — all commands, flags, and examples
- **API Reference (`api_reference.md`)** — Python API documentation for library use
- **Design Decisions (`design_decisions.md`)** — architectural rationale and trade-offs
- **Troubleshooting (`troubleshooting.md`)** — common issues and solutions

The nav section in `zensical.toml` is commented out until pages are written.

## Real-audio transcription mode

**Status:** Not yet started.

Currently Sonitra only transcribes synthesized audio rendered from MIDI. A planned mode
would feed a dataset's original recordings directly into the transcription step, bypassing
the render pipeline — enabling benchmarking on real instrument audio, which is the
canonical AMT evaluation scenario in the literature. For MAESTRO V3.0.0 this means using
the paired `.flac` recordings rather than re-synthesizing from MIDI.

Planned design:
- Pairing logic reads the dataset metadata CSV (e.g. `maestro-v3.0.0.csv`) to map each
  reference MIDI to its original `.flac`/`.wav` recording.
- The render step is skipped; transcription output is scored against the reference MIDI
  exactly as in the current workflow.
- Requires the full dataset audio download (MAESTRO V3.0.0 audio is ~120 GB); the
  MIDI-only zip (~57 MB) fetched by `scripts/download_datasets.py` is sufficient for the
  synthesize-then-transcribe workflow.
- This mode is distinct from the current pipeline where rendering fidelity is part of
  what is being benchmarked.

BSED (see "Additional datasets and instruments" above) is a simpler first target for
this mode than MAESTRO: `scripts/download_datasets.py bsed` already fetches real
recordings into `corpus/bsed/recordings/` alongside `corpus/bsed/midi/`, and the two
share a common `BSED-<NN>_...` filename prefix — pairing a score to its recordings is
a filename match, no metadata CSV lookup required.

## Note-level performance-alignment annotations (BSED)

**Status:** Not yet started.

BSED ships manually-verified note-level alignments between each score and its real
recordings under `03_NoteAnnotations/Note-Level-Alignment/` (CSV + `.npz`), plus a
less-precise `Sequence-Alignment/` variant — neither is currently fetched by
`scripts/download_datasets.py`. These map each score note to its actual onset/offset
in a specific real recording, a more precise reference than the nominal MIDI-score
timing the evaluation pipeline uses today, since real performances deviate from the
score (rubato, systematic delays, etc.).

Planned use: once real-audio transcription mode (above) exists, evaluation could
optionally use the per-recording aligned annotation as ground truth instead of raw
score MIDI, giving a tighter accuracy measurement for orchestral transcription. This
would require extending the `bsed` entry's `extract_map` in
`scripts/download_datasets.py` with an `annotations/` target, and adding an
alignment-format reference loader under `evaluation/` that consumes the CSV/npz
alignment format instead of parsing MIDI directly.
