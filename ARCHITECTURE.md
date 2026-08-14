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
        fx["Effects chain + normalisation"]
        audio[("Audio")]
        midi --> synth --> fx --> audio
        recordings --> fx
    end

    subgraph process["Process"]
        direction TB
        sep["Stem separation (optional) (Demucs · Passthrough)"]
        tx["Transcription (Basic Pitch · Precomputed · External)"]
        eval["Evaluation (Note · Frame · Expressive · DTW)"]
        audio --> sep --> tx --> eval
    end

    results[("Results (JSONL · summary.json)")]

    cfg --> render
    cfg --> process
    midi --> eval
    eval --> results
```

`render_pipeline.input_type: midi | audio` selects the render input: MIDI renders via synthesis, audio reads the source recording directly (synth skipped). Evaluation always uses the reference MIDIs in `midi/`.

## Render path (per file)

```mermaid
flowchart LR
    midi[("MIDI file")]
    parse["1. parse_midi (note dicts)"]
    tempo["2. Tempo rescale (native BPM → render_pipeline.bpm)"]
    synth["3. synth.render (per-thread backend)"]
    audio_in[("Audio file (recordings/)")]
    read["read_audio"]
    pre["Pre-normalise (only if pre_effects)"]
    fx["Effects chain (pedalboard)"]
    post["Post-normalise"]
    gate{"Quality gate (silence · clip · too short)"}
    write["write_audio (wav / flac / mp3)"]
    manifest[("Manifest (renders.jsonl + .failed.txt)")]

    midi --> parse --> tempo --> synth --> pre
    audio_in --> read --> pre
    pre --> fx --> post --> gate
    gate -- pass --> write --> manifest
    gate -- fail --> manifest
```

In audio mode the synth stage is skipped: `read_audio` replaces `parse_midi` → tempo rescale → `synth.render`. Audio-mode manifest entries record `source_path` (the source recording) alongside the usual fields.

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
    p["pipeline (synth_backend · effects_chain · input_type · bpm · sample_rate · max_workers)"]
    io["io (corpus_root · output_format · file_naming · dataset)"]
    dd["dawdreamer (block_size · plugin_path · faust_code)"]
    fs["fluidsynth (soundfont_path)"]
    pb["pedalboard (instrument · effects[])"]
    norm["normalisation (enabled · mode peak/rms · target_db · pre_effects)"]
    qg["quality_gates (silence · min_duration · clip)"]
    obs["observability (manifest · sse · progress)"]
    sep["separation (enabled · backend · model)"]
    tx["transcription (transcribers[] · max_workers)"]
    ev["evaluation (note · frame · expressive · dtw)"]
    bm["benchmark (conditions[] · sweeps[] · resume · max_workers)"]

    cfg --> p & io & dd & fs & pb & norm & qg & obs & sep & tx & ev & bm
```

Validators: `synth_backend=fluidsynth` requires `fluidsynth.soundfont_path`; `dawdreamer_vst` requires `dawdreamer.plugin_path`; `dawdreamer_faust` must not set it. These apply in MIDI mode only; with `render_pipeline.input_type: audio` the synth is never used, so the backend-field requirements are skipped. `validate_worker_constraint()` forces `max_workers=1` for DawDreamer `synth_backend`s regardless of `input_type` — it keys purely on `synth_backend`, so it still applies (harmlessly, since the synth is unused) if a DawDreamer backend is left configured in an audio-mode config.

### Corpus layout

```mermaid
flowchart LR
    root["{corpus}/{dataset}"]
    midi["midi/ (reference MIDIs)"]
    recordings["recordings/ (source audio)"]
    audio["audio/{config}/ (rendered audio)"]
    tx["transcription/{config}/ (output MIDIs)"]
    ev["eval_results/ (metrics)"]
    root --> midi & recordings & audio & tx & ev
```

`midi/` and `recordings/` are read-only sources; everything the pipeline writes lives under the work dir (`audio/{config}/`, `transcription/{config}/`, `eval_results/`). Audio files pair to reference MIDIs by token-prefix match (e.g. `BSED-01_1_*.wav` → `BSED-01_*.mid`) via `sonitra.corpus.pair_audio_to_reference` (top-down k-descent over `_`-split filename tokens; ambiguous or unmatched recordings are excluded, logged, and reported in `PairingResult`). `sonitra.corpus.discover_midi_files` / `discover_audio_files` do the recursive directory walks.

## Benchmark orchestration

```mermaid
flowchart TB
    bm["benchmark section (conditions[] · sweeps[])"]
    expand["expand_conditions (baseline → explicit conditions → one per sweep value; one-factor-at-a-time, no cross-product)"]
    conds["Condition list (name, dotted-path overrides)"]
    bm --> expand --> conds

    subgraph percond["Per condition"]
        direction TB
        apply["apply_overrides (fresh validated config)"]
        render["Re-render corpus (MIDI → synth, or recordings → effects)"]
        tx["Transcribe (enabled transcribers)"]
        eval["Evaluate (vs reference MIDI from midi/)"]
        apply --> render --> tx --> eval
    end

    conds --> percond
    percond --> records[("records JSONL (condition × file × transcriber)")]
    records --> summary[("summary.json (summary + degradation vs baseline)")]
    resume["Resume (config fingerprint; mismatch raises; audio mode keys on audio path)"]
    resume -.-> records
```

In audio mode the corpus recordings are the inputs, paired to reference MIDIs in `midi/`; DTW is skipped since real recordings cannot be compared against synth re-synthesis. Conditions/sweeps may not override `render_pipeline.input_type` (`_validate_no_input_type_sweep`, checked unconditionally before expansion) — input mode selects the corpus/pairing for the whole run and cannot vary per condition.

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

    audio[("Rendered audio")]
    est2[("Estimate notes")]
    resynth["Re-synthesis (same synth config)"]
    dtw["DTW (audio metric)"]

    audio --> dtw
    est2 --> resynth --> dtw
    dtw --> flat
```

DTW compares rendered audio against a re-synthesis of the transcription; it is skipped for audio input (real recordings).

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

    subgraph api["API (FastAPI)"]
        direction TB
        app["create_app"]
        store["JobStore"]
        worker["Render worker (asyncio.Lock; DawDreamer not concurrency-safe)"]
        cfgapi["PUT /config"]
        sse["SSE status stream"]
        routers["routers: jobs · health · config · status"]
    end

    subgraph scripts["Scripts"]
        direction TB
        batch["run_transcribe_eval.py (render → transcribe → evaluate per config)"]
        dl["download_datasets.py (corpora → midi/ + recordings/)"]
    end
```

`render` and `benchmark` read `render_pipeline.input_type` to select MIDI vs audio sources; `transcribe` and `evaluate` already accept `--audio` / `--reference` / `--estimate` path overrides. `--corpus` is asymmetric between the two commands: for `render` it means the recordings dir in audio mode (defaulting to `paths.recordings`), for `benchmark` it always means the reference-MIDI dir — recordings there come from `paths.recordings` unconditionally.

## Dependencies

```mermaid
flowchart TB
    core["Core deps (dawdreamer · mido · numpy · scipy · pedalboard · fastapi · uvicorn · httpx · typer · pyyaml · pydantic · rich · basic-pitch >=0.4,<0.5)"]
    ext["External (not pip): fluidsynth CLI · SoundFont (.sf2) · VST3 plugins"]
    extras["Optional extras"]
    demucs["[demucs] stem separation"]
    gpu["[gpu] TF GPU inference (Linux x86_64)"]
    dev["[dev] pytest tooling"]
    bp["[basicpitch] alias (already core)"]

    extras --> demucs & gpu & dev & bp
```

GPU: set `device: GPU:0` on a `basic_pitch` transcriber (default `cpu`). Docker GPU passthrough is a Compose profile (`--profile gpu`, service `sonitra-gpu`).

## Concurrency & testing

```mermaid
flowchart TB
    subgraph conc["Concurrency"]
        direction TB
        p1["render_pipeline.max_workers (parallel only for pedalboard_instrument; DawDreamer forced to 1)"]
        p2["benchmark.max_workers (ProcessPoolExecutor + worker event queue)"]
        p3["GPU: basic-pitch preallocates ~22 GB (keep transcription workers low)"]
    end

    subgraph test["Testing"]
        direction TB
        gate["pytest is the quality gate (no ruff/black/mypy)"]
        m1["skip_if_no_vst / integration (require VST_PATH / VST3_PATH)"]
        m2["slow (heavy backends: basic-pitch)"]
    end
```

Audio mode never touches the synth, so it parallelises freely regardless of `synth_backend`.