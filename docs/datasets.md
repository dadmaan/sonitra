# Datasets

Sonitra includes a standalone download script for standard AMT benchmark datasets:

```bash
python scripts/download_datasets.py --list             # show available datasets
python scripts/download_datasets.py                    # interactive picker (rich table)
python scripts/download_datasets.py maestro-v3-midi     # download MAESTRO V3.0.0 MIDI + metadata (~57 MB)
python scripts/download_datasets.py bsed                # download BSED MIDI + real recordings (~380 MB)
python scripts/download_datasets.py --all               # download everything (includes multi-GB entries, see below)
python scripts/download_datasets.py --all --jobs 4      # download everything, 4 at a time
python scripts/download_datasets.py maestro-v3-midi --output-dir /data/corpus  # custom path
```

Running the script with no dataset name opens an interactive picker: a rich table listing each dataset (number, key, name, size, target path, and present/missing status) with a prompt accepting comma-separated numbers, `all`, or `q` to quit. On a non-TTY or when `rich` is unavailable, the previous error message is shown instead. `--jobs N` downloads up to N selected datasets concurrently (default 1 = serial).

The script is stdlib-only (no venv required) and idempotent: re-running it skips datasets that are already present. A dataset entry can pull from more than one URL (e.g. MusicNet's MIDI, audio, and metadata are three separate files) and supports both `.zip` and `.tar.gz` archives.

Currently supported:

| Key | Dataset | Files | Size |
|---|---|---|---|
| `maestro-v3-midi` | [MAESTRO V3.0.0](https://magenta.withgoogle.com/datasets/maestro) — CC BY-NC-SA 4.0, noncommercial | 1,276 piano MIDI files + metadata, no audio | ~57 MB |
| `maestro-v3-wav` | MAESTRO V3.0.0 — CC BY-NC-SA 4.0, noncommercial | Paired recordings + metadata, no MIDI (downloads the same full archive as `-full` and discards MIDI members — MAESTRO ships no audio-only archive) | ~120 GB |
| `maestro-v3-full` | MAESTRO V3.0.0 — CC BY-NC-SA 4.0, noncommercial | MIDI + recordings + metadata | ~120 GB |
| `bsed` | [Beethoven Symphony Excerpt Dataset (BSED)](https://zenodo.org/records/20344500) v1.0 — CC BY-NC-SA 4.0, noncommercial | 20 MIDI scores + 100 real/synthetic recordings (5 per excerpt, pitch-corrected to A440) | ~380 MB |
| `musicnet` | [MusicNet](https://zenodo.org/records/5120004) — CC BY 4.0 | 330 classical recordings + reference MIDI + per-note label CSVs + track metadata | ~11 GB |
| `e-gmd-midi` | [Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd) — CC BY 4.0 | 45,537 drum performances (MIDI) + metadata, no audio | ~103 MB |
| `e-gmd-full` | Expanded Groove MIDI Dataset — CC BY 4.0 | MIDI + recordings + metadata | ~90 GB |

E-GMD is a drum-performance dataset — download-only for now, since Sonitra's transcription/evaluation backends target pitched instruments rather than drum-hit classification. [MAPS](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/) is not scripted: it's gated behind a registration form with no direct download URL, so it isn't a fit for this script's unattended download model.

Downloaded files land under `corpus/{dataset}/midi/` following the dataset-first layout (e.g. `corpus/maestro-v3/midi/2004/…`). Datasets that also ship real audio (e.g. `bsed`, `maestro-v3-full`/`-wav`, `musicnet`, `e-gmd-full`) additionally populate `corpus/{dataset}/recordings/` — deliberately not `audio/`, which is reserved for the pipeline's own rendered output (`corpus/{dataset}/audio/<config_name>/`). Datasets with descriptive/track-level metadata (CSV, JSON, README, LICENSE) populate `corpus/{dataset}/metadata/`. Additional datasets and instrument types are planned for future releases.

Note: if you downloaded `maestro-v3` with a version of this script prior to the `-midi`/`-wav`/`-full` split, its metadata files (`maestro-v3.0.0.csv`/`.json`, `README`, `LICENSE`) landed inside `midi/` rather than `metadata/`. Re-running `maestro-v3-midi` will treat `metadata/` as missing and re-fetch the (small) MIDI zip; the old files under `midi/` are unaffected and can be moved into `metadata/` by hand if desired.

---
[← Back to README](../README.md)
