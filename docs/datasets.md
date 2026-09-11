# Datasets

Sonitra ships a download script for standard test sets. AMT means automatic music transcription, turning audio into notes. A benchmark dataset is a shared set of scores and recordings you test against.

```bash
python scripts/download_datasets.py --list             # show available datasets
python scripts/download_datasets.py --notes            # show each dataset's instrument, contents, and intended task
python scripts/download_datasets.py                    # interactive picker (rich table)
python scripts/download_datasets.py maestro-v3-midi     # download MAESTRO V3.0.0 MIDI + metadata (~57 MB)
python scripts/download_datasets.py bsed                # download BSED MIDI + real recordings (~380 MB)
python scripts/download_datasets.py --all               # download everything (includes multi-GB entries, see below)
python scripts/download_datasets.py --all --jobs 4      # download everything, 4 at a time
python scripts/download_datasets.py maestro-v3-midi --output-dir /data/corpus  # custom path
```

If you run the script with no dataset name, it opens a picker. The picker shows a table with each dataset, its number, key, name, size, target path, and whether you already have it. Type comma-separated numbers to pick, `all` for all, or `q` to quit. Type `n` to switch to the notes page. It has one row per dataset, giving the instrument, what the set holds, and the task it was made for, with the same numbers as the main table. Press Enter to go back. If your terminal cannot show the table or `rich` is missing, you see an error message instead. `--jobs N` downloads up to N datasets at once. The default is 1, which means one after another.

The script uses only the Python standard library, so you do not need the project env to run it. You can run it again safely. It skips datasets you already have. One entry can pull from more than one web address. For example, MusicNet keeps its MIDI, audio, and metadata in three separate files. The script opens both `.zip` and `.tar.gz` files.

Have your own scores or recordings instead? See [Using your own dataset](custom-datasets.md).

Currently supported:

| Key | Dataset | Files | Size |
|---|---|---|---|
| `maestro-v3-midi` | [MAESTRO V3.0.0](https://magenta.withgoogle.com/datasets/maestro) — CC BY-NC-SA 4.0, noncommercial | 1,276 piano MIDI files + metadata, no audio | ~57 MB |
| `maestro-v3-wav` | MAESTRO V3.0.0 — CC BY-NC-SA 4.0, noncommercial | Paired recordings + metadata, no MIDI (downloads the same full archive as `-full` and discards MIDI members — MAESTRO ships no audio-only archive) | ~120 GB |
| `maestro-v3-full` | MAESTRO V3.0.0 — CC BY-NC-SA 4.0, noncommercial | MIDI + recordings + metadata | ~120 GB |
| `bsed` | [Beethoven Symphony Excerpt Dataset (BSED)](https://zenodo.org/records/20344500) v1.0 — CC BY-NC-SA 4.0, noncommercial | 20 MIDI scores + 100 real/synthetic recordings (5 per excerpt, pitch-corrected to A440) | ~380 MB |
| `musicnet` | [MusicNet](https://zenodo.org/records/5120004) — CC BY 4.0 | 330 classical recordings + reference MIDI + per-note label CSVs + track metadata | ~11 GB |
| `e-gmd-midi` | [Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd) — CC BY 4.0 | 1,059 drum performances, each replayed through 43 kits: 45,537 MIDI files + metadata, no audio | ~103 MB |
| `e-gmd-full` | Expanded Groove MIDI Dataset — CC BY 4.0 | MIDI + recordings + metadata | ~90 GB |
| `guitarset-mic` | [GuitarSet](https://zenodo.org/records/3371780) (Xi et al., ISMIR 2018) — CC BY 4.0 | 360 mono-mic recordings + JAMS annotations (requires conversion to MIDI, see below) | ~665 MB |
| `guitarset-mix` | GuitarSet (Xi et al., ISMIR 2018) — CC BY 4.0 | 360 pickup-mix recordings + JAMS annotations (requires conversion; shares `corpus/guitarset/` with `-mic`) | ~690 MB |
| `guitarset-full` | GuitarSet (Xi et al., ISMIR 2018) — CC BY 4.0 | Both recording variants (720 WAVs) + JAMS annotations in one run (requires conversion; one-run equivalent of `-mic` + `-mix`) | ~1.3 GB |
| `gaps-midi` | [GAPS](https://huggingface.co/datasets/xavriley/GAPS) (Riley et al., ISMIR 2024) — CC BY-NC-SA 4.0, research use | 404 classical-guitar MIDI references + metadata, no audio | ~3 MB |
| `gaps-full` | GAPS (Riley et al., ISMIR 2024) — CC BY-NC-SA 4.0, research use | 404 classical-guitar recordings (~23 h) + aligned MIDI + MusicXML + syncpoints + metadata, no conversion needed (re-uses MIDI already fetched by `gaps-midi`) | ~15.3 GB |

E-GMD holds drum performances, so you can download it but not yet benchmark it. Sonitra's transcription and scoring tools target pitched instruments like piano and guitar, not drum hits. [MAPS](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/) is not scripted. It sits behind a sign-up form with no direct download link, so the script cannot fetch it on its own.

Files go under `corpus/{dataset}/midi/`. This is the dataset-first layout, for example `corpus/maestro-v3/midi/2004/…`. MIDI means a digital score file. Sets that also ship real audio, such as `bsed`, `maestro-v3-full` and `-wav`, `musicnet`, and `e-gmd-full`, also fill `corpus/{dataset}/recordings/`. Sonitra keeps this separate on purpose from `audio/`, which holds only audio Sonitra itself renders (`corpus/{dataset}/audio/<config_name>/`). Sets with track notes (CSV, JSON, README, LICENSE files) fill `corpus/{dataset}/metadata/`. CSV means comma-separated values, a simple table format. Sonitra plans to add more sets and instruments later.

Note: if you fetched `maestro-v3` with an older script before the `-midi`/`-wav`/`-full` split, its metadata files (`maestro-v3.0.0.csv`/`.json`, `README`, `LICENSE`) landed in `midi/` instead of `metadata/`. If you run `maestro-v3-midi` again, the script sees `metadata/` as missing and fetches the small MIDI zip again. Your old files in `midi/` stay as they are. You can move them to `metadata/` by hand if you want.

### GuitarSet (real guitar audio, JAMS ground truth)

[GuitarSet](https://zenodo.org/records/3371780) (Xi et al., ISMIR 2018, CC BY 4.0) holds 360 real acoustic-guitar clips. Each is about 30 seconds, about 3 hours in total, with note-level ground truth. Ground truth means the correct answer you score against. JAMS is a music-label file format. This is the first guitar set in the script and the first whose answers arrive as JAMS instead of MIDI. The three keys share one folder (`corpus_subdir: "guitarset"`, the same pattern as the three `maestro-v3-*` variants). Fetch `guitarset-full` for both recording types in one run. You get 720 recordings in one `recordings/` folder. The mic-vs-pickup-mix difference shows in `_mic` vs `_mix` filename endings. Or fetch `guitarset-mic` or `guitarset-mix` alone for a usable single set. The 6-channel hex-pickup stems are deferred (see `ROADMAP.md`). A stem here means a separate pickup track.

You must convert the answers before use. The downloader puts nothing in `midi/` on purpose. It uses only the standard library, while the converter needs the project env. After each GuitarSet download, the script prints the next steps. Run the conversion once no matter which key you fetched:

```bash
python scripts/download_datasets.py guitarset-full   # JAMS → corpus/guitarset/annotations/, both audios → corpus/guitarset/recordings/
python scripts/guitarset_jams_to_midi.py --dry-run  # inspect counts first
python scripts/guitarset_jams_to_midi.py            # annotations/ → midi/ (360 unsuffixed stems) + metadata/guitarset.csv
sonitra benchmark --config config/benchmark/guitarset_test.yaml --dataset guitarset --limit 2
```

`scripts/guitarset_jams_to_midi.py` (standard library plus `sonitra.midi_writer`) merges the six per-string `note_midi` blocks into one note list per clip. It rounds pitch with `pitch = round(value)` and sets loudness to a fixed velocity of 100, because GuitarSet has no loudness data. It writes a General MIDI `program_change` on channel 0 ahead of the first note (`--program`, default 24, Acoustic Guitar (nylon); `--no-program` omits it): without one, a GM player such as the Windows GS Wavetable synth falls back to program 0 and the reference plays back as piano. The program affects playback timbre only — `parse_midi` reads notes and ignores it, so no metric changes. It writes `corpus/guitarset/midi/*.mid` plus `corpus/guitarset/metadata/guitarset.csv`, one row per clip with join column `midi_filename`. It also writes a `<csv>.provenance.json` log file. `annotations/` is a new corpus target folder, the same one `ROADMAP.md` plans for BSED's alignment files.

Use GuitarSet with `sonitra benchmark` only. `sonitra evaluate` and `scripts/run_transcribe_eval.py` match files by exact relative path, so they cannot match the suffixed audio stems (`*_mic`/`*_mix`) to their plain references. `sonitra benchmark` can, because its token-prefix matching handles the suffix. See `.local/notes/TODO/guitarset.md` for the full caveat list (fixed loudness, absent `dtw.*`, unison false negatives, semitone rounding, MIDI-mode `bpm != 120` warning, upstream errata).

### GAPS (long-form classical guitar, aligned MIDI ground truth)

[GAPS](https://huggingface.co/datasets/xavriley/GAPS) (Guitar-Aligned Performance Scores; Riley et al., ISMIR 2024) holds 404 classical-guitar recordings from more than 200 players, about 23 hours in all. The paper's 14 hours covers only the 300 recordings in the published split. The recordings were made in every kind of place, from professional studios to phone microphones, which is exactly the range the benchmark exists to measure. The middle piece runs about 3 minutes and the longest about 17 minutes, so these are whole works rather than the short clips GuitarSet ships.

Nothing needs converting. The ground truth arrives as `.mid` already lined up with the audio timeline, and the file stems match across folders (`001_mvswc.wav` pairs with `001_mvswc.mid`). Pitches are the notes that sound, not the notes as written, so no octave correction is needed either. Loudness is a fixed velocity of 100 throughout, because GAPS carries no dynamics.

GAPS is the first entry the script fetches from Hugging Face rather than from one archive file. Hugging Face serves it as 1,617 separate files with no zip to download, so the script has a source kind for this shape, `hf_tree`. It lists a folder through the Hugging Face tree API, then fetches each file. The version is pinned to one commit (`b4c89a33…`), never a moving branch, so a later upstream release cannot change the corpus under a published benchmark. Re-running the download skips any file already on disk at the listed size and resumes a half-finished one, so an interrupted fetch costs you only what it had not yet finished. `--force` re-fetches everything from scratch.

Two keys share `corpus/gaps/`, the same pattern as the `maestro-v3-*` and `e-gmd-*` pairs. `gaps-midi` fetches only the 404 MIDI references and the metadata CSV, about 3 MB. That is all a MIDI-input run needs, because Sonitra makes its own audio from the MIDI. `gaps-full` adds the real recordings, MusicXML, and syncpoints, about 15.3 GB. Audio-input runs need it, and it also covers every MIDI-input run. The two keys fetch the same pinned commit, so you can start with `gaps-midi` and add `gaps-full` later. `gaps-full` then checks each MIDI file and skips it when it is already on disk at the listed size. Any file that is missing or cut short is downloaded. With no MIDI on disk, `gaps-full` downloads everything. The script prints one line per folder, such as `[gaps-full] midi: 404 already present, 0 downloaded`, so you can see what it re-used.

```bash
python scripts/download_datasets.py gaps-midi   # ~3 MB: MIDI + metadata into corpus/gaps/
python scripts/download_datasets.py gaps-full   # ~15.3 GB: adds recordings, MusicXML, syncpoints
sonitra benchmark --config config/benchmark/gaps_test.yaml --dataset gaps --limit 2
```

The download key and the dataset name differ. `--dataset gaps` stays the same whichever key you fetched, because both keys fill `corpus/gaps/`.

Files land in `corpus/gaps/recordings/` (audio, `gaps-full` only), `corpus/gaps/midi/`, `corpus/gaps/annotations/musicxml/` and `corpus/gaps/annotations/syncpoints/` (`gaps-full` only), and `corpus/gaps/metadata/`. Audio goes to `recordings/`, never `audio/`, which stays reserved for Sonitra's own renders.

Sonitra keeps all 404 recordings, unfiltered. Published GAPS is the 300 files with a non-empty `split` column, and a separate rule of `f-measure >= 0.75` also picks 300 files, but the two sets overlap in only 250. They measure different things. `f-measure` records how well an existing transcription model agreed with the score, so filtering on it would delete the hardest-sounding recordings first, which are the ones a degradation study most needs. Both columns reach a benchmark export as `meta.split` and `meta.f-measure`, so you can filter afterwards if you want to.

Three recordings (`Bc1wc`, `CM1wc`, `mN1wc`) are upstream leftovers. They lack the numeric filename prefix, use a different name ending in every folder, and are absent from the metadata CSV. They never pair, so `sonitra benchmark` logs one warning each and moves on. They stay on disk. In MIDI-input mode there is nothing to pair: their MIDI files (`*-fine-aligned.mid`) render like any other and only their `meta.*` columns come out blank.

Use GAPS with `sonitra benchmark`. `sonitra evaluate` and `scripts/run_transcribe_eval.py` would in fact work, because the stems match exactly, but wiring them up is deferred (see `ROADMAP.md`). Read `.local/notes/TODO/gaps.md` before reporting numbers: the ground truth is a score aligned to a performance rather than a transcription of one, so note ends and lengths are the written ones. On guitar, notes ring well past their written length, which makes offset and frame scores read pessimistically. A perfect transcriber cannot reach 1.0 here. The benchmark reports differences against the baseline condition, and a constant bias in the ground truth cancels out in a difference, which is what makes the set usable.

### Joining dataset metadata into a benchmark export

`scripts/export_regression_table.py` (see [CLI reference](cli.md)) can add a dataset's `corpus/{dataset}/metadata/*.csv` to a benchmark results table. Use this to add composer or work details for further analysis, for example:

```bash
python scripts/export_regression_table.py \
  --work-dir corpus/maestro-v3/benchmark/vintage_scenarios_MIDI_INPUT \
  --metadata-csv corpus/maestro-v3/metadata/maestro-v3.0.0.csv \
  --metadata-join-column midi_filename
```

The joined table is the input that `scripts/run_mixed_effects_analysis.py` expects (see [Statistical analysis](statistical-analysis.md)).

The join works for any dataset. `--metadata-join-column` names the CSV column that holds a filename. MAESTRO's column is `midi_filename`. MusicNet's or a future set's column may differ. Sonitra matches it against each benchmark row by file name only. Every other column from a matched row is added as `meta.<column>`. Sonitra makes no demand that sets share the same columns, because they do not. For example, MusicNet's metadata has `movement` and `ensemble`, while MAESTRO's does not.

---
[← Back to README](../README.md)
