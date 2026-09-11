# Architecture

## System overview

```mermaid
flowchart TB
    cfg["PipelineConfig (config/*.yaml)"]

    subgraph render["Render"]
        direction TB
        midi[("MIDI corpus (midi/)")]
        recordings[("Audio corpus (recordings/)")]
        synth["Synthesis (FluidSynth · DawDreamer · PedalboardSynth)"]
        fx["Normalisation + effects chain"]
        audio[("Audio")]
        midi --> synth --> fx --> audio
        recordings --> fx
    end

    subgraph process["Process"]
        direction TB
        sep["Stem separation (optional, benchmark only) (Demucs · Passthrough)"]
        tx["Transcription (Basic Pitch · Precomputed · External)"]
        eval["Evaluation (Note · Frame · Expressive · DTW)"]
        audio --> tx
        audio -.-> sep -.-> tx
        tx --> eval
    end

    results[("Results (JSONL · summary.json)")]

    cfg --> render
    cfg --> process
    midi --> eval
    eval --> results
```

`render_pipeline.input_type: midi | audio` selects the render input: MIDI renders via synthesis, audio reads the source recording directly (synth skipped). Evaluation always uses the reference MIDIs in `midi/`. Stem separation and DTW run only inside `sonitra benchmark`; the standalone `transcribe` / `evaluate` commands skip them.

## Render path (per file)

```mermaid
flowchart LR
    src[("Source file (MIDI or recording)")]
    skip{"Output exists and overwrite: false?"}
    skipped(["skipped (no manifest entry)"])
    parse["parse_midi (notes in file-tempo seconds · programs)"]
    synth["synth.render(notes, program) (per-thread backend)"]
    read["read_audio (native sample rate)"]
    pre["Pre-normalise (if enabled and pre_effects)"]
    fx["Effects chain (if effects_chain: pedalboard)"]
    post["Post-normalise (if enabled and not pre_effects)"]
    gate{"Quality gate (silence · clip · too short)"}
    write["write_audio (wav / flac / mp3)"]
    manifest[("Manifest (renders.jsonl · .failed.txt)")]
    err["Exception at any stage"]

    src --> skip
    skip -- yes --> skipped
    skip -- "no, MIDI" --> parse --> synth --> pre
    skip -- "no, audio" --> read --> pre
    pre --> fx --> post --> gate
    gate -- pass --> write --> manifest
    gate -- fail --> manifest
    err -.-> manifest
```

- **Tempo**: notes follow the MIDI file's own tempo map; `render_pipeline.bpm` is host tempo only (DawDreamer `set_bpm`, FluidSynth tick grid).
- **Program**: a file with exactly one program passes it to the synth; several programs pass none (warned once). FluidSynth uses `fluidsynth.program` if set, else the file program, else the SoundFont default; DawDreamer and Pedalboard ignore it.
- **Normalisation** runs at most once per file: before effects if `pre_effects`, otherwise after.
- **Fail-soft**: gate failures and exceptions are recorded and the batch continues. The effects chain is built before the loop, so a chain that fails to load aborts the batch.
- **Manifest** is opt-in (`observability.write_manifest`, `write_failed_list`). Audio-mode entries record `source_path` (the recording) alongside the usual fields.

## Pluggable-backend idiom

```mermaid
flowchart TB
    subgraph idiom["Three-part shape"]
        direction LR
        proto["Protocol (runtime_checkable)"]
        reg["Registry (discriminator → builder)"]
        factory["make_* factory (lazy import)"]
        inst["Backend instance"]
        proto --> factory
        reg --> factory
        factory --> inst
    end

    subgraph families["Where it applies"]
        direction TB
        t1["Transcribers (register_transcriber → make_transcriber)"]
        s1["Separators (register_separator → make_separator)"]
        m1["Metrics (register_symbolic_metric / register_audio_metric)"]
        x1["Synthesisers: exception (make_synth = if/elif dispatch over SynthBackend, no registry)"]
        x2["Sources: exception (make_source = if/elif dispatch over InputType, no registry; MidiSource wraps parse_midi+make_synth, AudioSource wraps read_audio)"]
    end
```

## Configuration

```mermaid
flowchart TB
    cfg["PipelineConfig (extra='forbid' everywhere)"]
    p["render_pipeline (input_type · synth_backend · effects_chain · bpm = host tempo · overwrite · max_workers)"]
    io["io (corpus_root · dataset · output_format)"]
    dd["dawdreamer (plugin_path · faust_code)"]
    fs["fluidsynth (soundfont_path · program)"]
    pb["pedalboard (instrument · effects[])"]
    norm["normalisation (enabled · pre_effects)"]
    qg["quality_gates (silence · clip · min_duration)"]
    obs["observability (write_manifest · write_failed_list)"]
    sep["separation (enabled · backend · stem)"]
    tx["transcription (transcribers[] · max_workers)"]
    ev["evaluation (note · frame · expressive · dtw · max_workers)"]
    bm["benchmark (conditions[] · sweeps[] · resume · save_audio · max_workers)"]

    cfg --> p & io & dd & fs & pb & norm & qg & obs & sep & tx & ev & bm
```

Only behaviour-driving fields are shown; `config/source.yaml` documents every key.

Validators apply in MIDI mode only (audio mode never builds a synth): `synth_backend=fluidsynth` requires `fluidsynth.soundfont_path`; `dawdreamer_vst` requires `dawdreamer.plugin_path`; `dawdreamer_faust` must not set it. `validate_worker_constraint()` keys only on `synth_backend`, so a DawDreamer backend forces `render_pipeline.max_workers=1` in either mode.

### Corpus layout

```mermaid
flowchart LR
    root["{corpus}/{dataset}"]
    midi["midi/ (reference MIDIs)"]
    recordings["recordings/ (source audio)"]
    meta["metadata/ (per-file CSV + provenance)"]
    ann["annotations/ (raw labels: JAMS, MusicNet CSVs, score MIDI)"]
    audio["audio/{config}/ (rendered audio)"]
    tx["transcription/{config}/ (output MIDIs)"]
    ev["eval_results/ (metrics)"]
    bm["benchmark/{config}/ (benchmark work dir)"]
    root --> midi & recordings & meta & ann & audio & tx & ev & bm
```

`midi/`, `recordings/`, `metadata/` and `annotations/` are produced by `scripts/download_datasets.py` (which also keeps download records in `.sources/`) and the dataset converters; the pipeline only reads them. The CLI writes `audio/{config}/`, `transcription/{config}/` and `eval_results/`; `sonitra benchmark --dataset` writes `benchmark/{config}/` (`./benchmark/{config}/` without `--dataset`).

Recordings pair to reference MIDIs by token prefix (e.g. `BSED-01_1_*.wav` → `BSED-01_*.mid`) via `sonitra.corpus.pair_audio_to_reference`: `k` descends over `_`-split stem tokens and stops at the first `k` with one candidate (paired) or several (ambiguous). Unmatched and ambiguous recordings are excluded, logged and reported in `PairingResult` (`ambiguous` keeps their candidates). `scripts/check_dataset.py` calls the same function; its core, `match_token_prefix`, also backs `export_regression_table.py --metadata-match token-prefix`. `discover_midi_files` / `discover_audio_files` do the recursive directory walks.

## Benchmark orchestration

```mermaid
flowchart TB
    bm["benchmark section (conditions[] · sweeps[])"]
    expand["expand_conditions (baseline → explicit conditions → one per sweep value; one factor at a time, no cross-product)"]
    conds["Condition list (name, dotted-path overrides)"]
    setup["Once per run: parse reference MIDIs · pair recordings (audio mode) · resume check · write config.yaml + .fingerprint"]
    bm --> expand --> conds

    subgraph percond["Per condition (benchmark.max_workers = processes)"]
        direction TB
        apply["apply_overrides (fresh validated config)"]
        render["run_pipeline → audio/{slug}/"]
        sep["Separate (optional) → stems/{slug}/"]
        tx["Transcribe (enabled transcribers) → transcriptions/{slug}/"]
        eval["Evaluate vs reference (+ DTW if enabled, MIDI mode)"]
        clean["Delete audio + stems if save_audio: false"]
        apply --> render --> tx
        render -.-> sep -.-> tx
        tx --> eval --> clean
    end

    conds --> percond
    setup --> percond
    percond --> records[("benchmark_results.jsonl (condition × source file × transcriber)")]
    records --> summary[("summary.json (summary · degradation vs baseline · timing + host)")]
    resume["Resume (skips completed cells; fingerprint mismatch raises)"]
    resume -.-> records
```

A failed render yields `render_failed` records for that file and the run continues. In audio mode the recordings are the inputs, paired once to reference MIDIs in `midi/`, and records key on the recording path; DTW is skipped. Conditions/sweeps may not override `render_pipeline.input_type` (`_validate_no_input_type_sweep`, checked before expansion): input mode selects the corpus and pairing for the whole run.

## Evaluation

```mermaid
flowchart TB
    ref[("Reference notes (parse_midi of source MIDI)")]
    est[("Estimate notes (transcription)")]

    subgraph sym["Symbolic metrics"]
        direction LR
        note["Note (onset · offset · velocity F1)"]
        frame["Frame"]
        expr["Expressive"]
    end

    ref --> sym
    est --> sym
    sym --> flat["'<metric>.<key>' values (NaN = undefined, skipped in aggregation)"]

    audio[("Rendered audio (post-effects)")]
    est2[("Estimate notes")]
    resynth["Re-synthesis (same synth config; no effects, normalisation or file program)"]
    dtw["DTW (audio metric; benchmark only)"]

    audio --> dtw
    est2 --> resynth --> dtw
    dtw --> flat
```

DTW runs only in `sonitra benchmark`, in MIDI mode, when `evaluation.dtw.enabled` (off by default). Its re-synthesis skips the effects chain, normalisation and the file's program, so the distance also reflects those differences, not only transcription error.

## Interfaces

```mermaid
flowchart TB
    subgraph cli["CLI (sonitra / python -m sonitra)"]
        direction LR
        init["init"]
        render["render"]
        tx["transcribe"]
        ev["evaluate"]
        bm["benchmark"]
        serve["serve"]
    end

    subgraph api["API (FastAPI, render only)"]
        direction TB
        app["create_app"]
        store["JobStore"]
        worker["Render worker (asyncio.Lock; DawDreamer not concurrency-safe)"]
        routers["routers: jobs · health/ready · config (GET/PUT) · status (SSE stream)"]
    end

    subgraph scripts["Scripts (standalone)"]
        direction TB
        dl["download_datasets.py (stdlib-only; → midi/ · recordings/ · annotations/ · metadata/)"]
        conv["guitarset_jams_to_midi.py · musicnet_labels_to_midi.py (annotations/ → midi/ + metadata/)"]
        check["check_dataset.py (pre-run pairing and MIDI checks)"]
        batch["run_transcribe_eval.py (render → transcribe → evaluate per preset)"]
        analysis["enrich_metadata.py → export_regression_table.py → run_mixed_effects_analysis.py (R)"]
    end
```

`render` and `benchmark` read `render_pipeline.input_type` to select MIDI vs audio sources; `transcribe` and `evaluate` accept `--audio` / `--reference` / `--estimate` path overrides. `--corpus` differs between the two: for `render` it is the recordings dir in audio mode (default `paths.recordings`); for `benchmark` it is always the reference-MIDI dir, and recordings always come from `paths.recordings`. API jobs without a config fall back to the legacy `engine=` render path. `sonitra.midi_writer` is shared by transcription output and the converters; `sonitra.unisons` by the converters.

## Dependencies

```mermaid
flowchart TB
    core["Core deps (dawdreamer · mido · numpy · scipy · pedalboard · fastapi · uvicorn · httpx · typer · pyyaml · pydantic · rich · basic-pitch >=0.4,<0.5)"]
    ext["External (not pip): fluidsynth CLI · SoundFont (.sf2) · VST3 plugins · R + glmmTMB · jsonlite"]
    extras["Optional extras"]
    demucs["[demucs] stem separation"]
    gpu["[gpu] CUDA wheels for TF GPU inference (bare-metal Linux x86_64)"]
    dev["[dev] pytest tooling"]
    bp["[basicpitch] alias (already core)"]

    extras --> demucs & gpu & dev & bp
```

GPU: set `device: GPU:0` on a `basic_pitch` transcriber (default `cpu`). Docker GPU passthrough is a Compose profile (`--profile gpu`, service `sonitra-gpu`; the GPU image installs CUDA itself rather than using the extra).

## Concurrency & testing

```mermaid
flowchart TB
    subgraph conc["Concurrency"]
        direction TB
        p1["render_pipeline.max_workers (threads only when synth_backend = pedalboard_instrument, in either input mode; DawDreamer forced to 1)"]
        p2["benchmark.max_workers (one process per condition + worker event queue)"]
        p3["transcription.max_workers · evaluation.max_workers (threads in CLI transcribe / evaluate)"]
    end

    subgraph test["Testing"]
        direction TB
        gate["pytest is the quality gate (no ruff/black/mypy)"]
        m1["skip_if_no_vst / integration (require VST_PATH / VST3_PATH)"]
        m2["slow (heavy backends: basic-pitch)"]
        m3["requires_r (R with glmmTMB)"]
    end
```

Render parallelism keys only on `synth_backend`, so audio mode renders serially unless `synth_backend: pedalboard_instrument`.
