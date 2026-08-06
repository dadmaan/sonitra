# Python API

```python
from sonitra.config import load_config, resolve_corpus_paths
from sonitra.pipeline import run_pipeline
from pathlib import Path

cfg = load_config("config/examples/pedalboard_baseline.yaml")
paths = resolve_corpus_paths(cfg, config_name="pedalboard_baseline")
# paths.midi          → corpus/test/midi
# paths.audio         → corpus/test/audio/pedalboard_baseline
# paths.transcription → corpus/test/transcription/pedalboard_baseline
# paths.eval_results  → corpus/test/eval_results

midi_files = sorted(
    p for p in paths.midi.rglob("*")
    if p.is_file() and p.suffix.lower() in {".mid", ".midi"}
)
result = run_pipeline(
    midi_files,
    out_dir=paths.audio,
    config=cfg,
)
print(f"Done: {result.succeeded}, Failed: {result.failed}")
```

---
[← Back to README](../README.md)
