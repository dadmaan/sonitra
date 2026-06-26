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

**Status:** Not yet started.

The current corpus targets piano. MAESTRO V3.0.0 MIDI files can be downloaded via
`scripts/download_datasets.py`. Planned expansion to cover additional instruments,
multi-instrument datasets (e.g. Slakh2100), and automated download support for
further datasets.

## Additional transcription backends

**Status:** Not yet started.

Beyond Basic Pitch and the generic `external_command` backend, native integrations with other AMT systems (e.g. MT3, Omnizart) are under consideration.

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
