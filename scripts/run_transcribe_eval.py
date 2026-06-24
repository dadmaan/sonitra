#!/usr/bin/env python3
"""
Transcribe + evaluate all configs in config/ against pre-rendered audio in corpus/audio/.

For each config:
  1. sonitra transcribe  -> corpus/transcription/<config>/<backend>/*.mid
  2. sonitra evaluate    -> corpus/eval_results/<config>.jsonl

After all configs:
  corpus/eval_results/summary.jsonl  -- one line per config, mean of per-file metrics

Usage:
    python scripts/run_transcribe_eval.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO / "config"
AUDIO_DIR = REPO / "corpus" / "audio"
TRANSCRIPTION_DIR = REPO / "corpus" / "transcription"
EVAL_DIR = REPO / "corpus" / "eval_results"
MIDI_REF_DIR = REPO / "corpus" / "midi"

PYTHON = sys.executable


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def _mean_metrics(rows: list[dict]) -> dict[str, float | None]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "stem"]
    result: dict[str, float | None] = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and not math.isnan(r[key])]
        result[key] = sum(vals) / len(vals) if vals else None
    return result


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # json.dumps does not support NaN; replace with null.
            fh.write(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}) + "\n")


def main() -> int:
    TRANSCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    if not configs:
        print(f"No configs found in {CONFIGS_DIR}", file=sys.stderr)
        return 1

    failures: list[str] = []
    summary_rows: list[dict] = []

    for config_path in configs:
        name = config_path.stem
        audio_dir = AUDIO_DIR / name

        bar = "=" * 60
        print(f"\n{bar}")
        print(f"CONFIG: {name}")

        if not audio_dir.exists():
            print(f"  SKIP — no audio dir at {audio_dir}")
            continue

        wavs = sorted(audio_dir.glob("*.wav"))
        if not wavs:
            print(f"  SKIP — no WAV files in {audio_dir}")
            continue

        print(f"  audio : {audio_dir}  ({len(wavs)} WAVs)")

        # ── Step 1: transcribe ──────────────────────────────────────────────
        transcription_out = TRANSCRIPTION_DIR / name
        print(f"  step 1: transcribe -> {transcription_out}/")
        rc = _run([
            PYTHON, "-m", "sonitra", "transcribe",
            "--config", str(config_path),
            "--audio",  str(audio_dir),
            "--output", str(transcription_out),
        ])
        if rc != 0:
            print(f"  FAIL  — transcribe exited {rc}")
            failures.append(f"{name}:transcribe")
            continue

        # ── Step 2: find backend subdir ─────────────────────────────────────
        estimate_dir = transcription_out / "basic_pitch"
        if not estimate_dir.exists():
            subdirs = [d for d in transcription_out.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"  FAIL  — no transcription output subdir found under {transcription_out}")
                failures.append(f"{name}:evaluate")
                continue
            estimate_dir = sorted(subdirs)[0]
        print(f"  midis : {estimate_dir}/")

        # ── Step 3: evaluate ────────────────────────────────────────────────
        eval_out = EVAL_DIR / f"{name}.jsonl"
        print(f"  step 2: evaluate   -> {eval_out}")
        rc = _run([
            PYTHON, "-m", "sonitra", "evaluate",
            "--config",    str(config_path),
            "--reference", str(MIDI_REF_DIR),
            "--estimate",  str(estimate_dir),
            "--output",    str(eval_out),
        ])
        if rc != 0:
            print(f"  FAIL  — evaluate exited {rc}")
            failures.append(f"{name}:evaluate")
            continue

        # ── Step 4: accumulate summary ──────────────────────────────────────
        rows = _load_jsonl(eval_out)
        means = _mean_metrics(rows)
        summary_rows.append({"config": name, **means})

        onset_f1_key = next((k for k in means if "onset_f1" in k), None)
        if onset_f1_key and means[onset_f1_key] is not None:
            print(f"  onset_f1 (mean): {means[onset_f1_key]:.3f}")

        print(f"  OK")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_path = EVAL_DIR / "summary.jsonl"
    _dump_jsonl(summary_rows, summary_path)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(summary_rows)} configs processed, {len(failures)} failed.")
    if failures:
        print("Failed steps:")
        for f in failures:
            print(f"  {f}")
    print(f"Summary -> {summary_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
