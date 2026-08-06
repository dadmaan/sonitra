# Datasets

Sonitra includes a standalone download script for standard AMT benchmark datasets:

```bash
python scripts/download_datasets.py --list          # show available datasets
python scripts/download_datasets.py maestro-v3         # download MAESTRO V3.0.0 MIDI (~57 MB)
python scripts/download_datasets.py bsed                # download BSED MIDI + real recordings (~380 MB)
python scripts/download_datasets.py --all           # download everything
python scripts/download_datasets.py maestro-v3 --output-dir /data/corpus  # custom path
```

The script is stdlib-only (no venv required) and idempotent: re-running it skips datasets that are already present.

Currently supported:

| Key | Dataset | Files |
|---|---|---|
| `maestro-v3` | [MAESTRO V3.0.0](https://magenta.withgoogle.com/datasets/maestro) MIDI-only | ~1,276 piano MIDI files |
| `bsed` | [Beethoven Symphony Excerpt Dataset (BSED)](https://zenodo.org/records/20344500) v1.0 — CC BY-NC-SA 4.0, noncommercial | 20 MIDI scores + 100 real/synthetic recordings (5 per excerpt, pitch-corrected to A440) |

Downloaded files land under `corpus/{dataset}/midi/` following the dataset-first layout (e.g. `corpus/maestro-v3/midi/2004/…`). Datasets that also ship real audio (e.g. `bsed`) additionally populate `corpus/{dataset}/recordings/` — deliberately not `audio/`, which is reserved for the pipeline's own rendered output (`corpus/{dataset}/audio/<config_name>/`). Additional datasets and instrument types are planned for future releases.

---
[← Back to README](../README.md)
