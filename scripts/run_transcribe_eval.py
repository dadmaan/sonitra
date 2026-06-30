#!/usr/bin/env python3
"""
Render, transcribe and evaluate all configs in config/ against the corpus.

For each config:
  1. sonitra render      -> corpus/{dataset}/audio/<config>/*.wav
  2. sonitra transcribe  -> corpus/{dataset}/transcription/<config>/<backend>/*.mid
  3. sonitra evaluate    -> corpus/{dataset}/eval_results/<config>.jsonl

After all configs:
  corpus/{dataset}/eval_results/summary.jsonl   -- one line per config, mean of per-file metrics
  corpus/{dataset}/eval_results/summary.csv     -- same data as summary.jsonl, CSV format
  corpus/{dataset}/eval_results/all_results.csv -- flat table: one row per (config, file)

Usage:
    python scripts/run_transcribe_eval.py
    python scripts/run_transcribe_eval.py --dataset maestro-v3
    python scripts/run_transcribe_eval.py --dataset test --skip-render
    python scripts/run_transcribe_eval.py --dataset maestro-v3 --limit 10
    python scripts/run_transcribe_eval.py --dataset maestro-v3 --limit 10 --seed 42
    python scripts/run_transcribe_eval.py --config pedalboard_baseline
    python scripts/run_transcribe_eval.py --config pedalboard_baseline pedalboard_no_effects
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO / "config/examples"

PYTHON = sys.executable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render, transcribe and evaluate all configs against the corpus."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="NAME",
        help=(
            "Optional dataset name.  When set, all corpus paths are scoped under "
            "corpus/{dataset}/midi|audio|transcription|eval_results/."
        ),
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        default=False,
        help="Skip the render step and use already-rendered audio.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Render at most N MIDI files per config (random subset; passed to sonitra render).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="RNG seed for --limit sampling (passed to sonitra render). Default: 123.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Run only the named config(s) (stem without .yaml). "
            "Omit to run all configs under config/."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of configs to process in parallel (default: 1). "
            "Each config's render→transcribe→evaluate steps still run serially."
        ),
    )
    return parser.parse_args()


def _resolve_dirs(dataset: str | None) -> tuple[Path, Path, Path, Path]:
    """Return (AUDIO_DIR, TRANSCRIPTION_DIR, EVAL_DIR, MIDI_REF_DIR).

    Args:
        dataset: Dataset name from ``--dataset``, or ``None`` for the default
            (non-scoped) layout.

    Returns:
        Four :class:`pathlib.Path` objects for the audio, transcription,
        evaluation results, and MIDI reference directories respectively.
        All paths follow the dataset-first layout:
        ``corpus/{dataset}/audio``, ``corpus/{dataset}/transcription``, etc.
    """
    root = REPO / "corpus"
    base = root / dataset if dataset is not None else root
    return (
        base / "audio",
        base / "transcription",
        base / "eval_results",
        base / "midi",
    )


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def _mean_metrics(rows: list[dict]) -> dict[str, float | None]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "file"]
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
            fh.write(
                json.dumps(
                    {
                        k: (None if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in row.items()
                    }
                )
                + "\n"
            )


def _dump_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if isinstance(v, float) and math.isnan(v)
                        else ("" if v is None else v)
                    )
                    for k, v in row.items()
                }
            )


def _process_config(
    config_path: Path,
    args: argparse.Namespace,
    audio_dir_base: Path,
    transcription_dir_base: Path,
    eval_dir: Path,
    midi_ref_dir: Path,
) -> tuple[list[dict], list[dict], list[str]]:
    """Run render → transcribe → evaluate for one config.

    Returns ``(summary_rows, all_rows, failures)`` so the caller can merge
    results across configs, including when running configs in parallel.
    """
    name = config_path.stem
    audio_dir = audio_dir_base / name
    summary_rows: list[dict] = []
    all_rows: list[dict] = []
    failures: list[str] = []

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"CONFIG: {name}")

    if not args.skip_render:
        print(f"  step 1: render     -> {audio_dir}/")
        render_cmd = [PYTHON, "-m", "sonitra", "render", "--config", str(config_path)]
        if args.dataset is not None:
            render_cmd += ["--dataset", args.dataset]
        if args.limit is not None:
            render_cmd += ["--limit", str(args.limit)]
        if args.seed is not None:
            render_cmd += ["--seed", str(args.seed)]
        rc = _run(render_cmd)
        if rc != 0:
            print(f"  FAIL  — render exited {rc}")
            failures.append(f"{name}:render")
            return summary_rows, all_rows, failures

    if not audio_dir.exists():
        print(f"  SKIP — no audio dir at {audio_dir}")
        return summary_rows, all_rows, failures

    wavs = sorted(audio_dir.rglob("*.wav"))
    if not wavs:
        print(f"  SKIP — no WAV files in {audio_dir}")
        return summary_rows, all_rows, failures

    print(f"  audio : {audio_dir}  ({len(wavs)} WAVs)")

    transcription_out = transcription_dir_base / name
    print(f"  step 2: transcribe -> {transcription_out}/")
    rc = _run(
        [
            PYTHON,
            "-m",
            "sonitra",
            "transcribe",
            "--config",
            str(config_path),
            "--audio",
            str(audio_dir),
            "--output",
            str(transcription_out),
        ]
    )
    if rc != 0:
        print(f"  FAIL  — transcribe exited {rc}")
        failures.append(f"{name}:transcribe")
        return summary_rows, all_rows, failures

    estimate_dir = transcription_out / "basic_pitch"
    if not estimate_dir.exists():
        subdirs = [d for d in transcription_out.iterdir() if d.is_dir()]
        if not subdirs:
            print(
                f"  FAIL  — no transcription output subdir found under {transcription_out}"
            )
            failures.append(f"{name}:evaluate")
            return summary_rows, all_rows, failures
        estimate_dir = sorted(subdirs)[0]
    print(f"  midis : {estimate_dir}/")

    eval_out = eval_dir / f"{name}.jsonl"
    print(f"  step 3: evaluate   -> {eval_out}")
    rc = _run(
        [
            PYTHON,
            "-m",
            "sonitra",
            "evaluate",
            "--config",
            str(config_path),
            "--reference",
            str(midi_ref_dir),
            "--estimate",
            str(estimate_dir),
            "--output",
            str(eval_out),
        ]
    )
    if rc != 0:
        print(f"  FAIL  — evaluate exited {rc}")
        failures.append(f"{name}:evaluate")
        return summary_rows, all_rows, failures

    rows = _load_jsonl(eval_out)
    means = _mean_metrics(rows)
    summary_rows.append({"config": name, **means})
    all_rows.extend({"config": name, **r} for r in rows)

    onset_f1_key = next((k for k in means if "onset_f1" in k), None)
    if onset_f1_key and means[onset_f1_key] is not None:
        print(f"  onset_f1 (mean): {means[onset_f1_key]:.3f}")
    print("  OK")
    return summary_rows, all_rows, failures


def main() -> int:
    args = _parse_args()
    AUDIO_DIR, TRANSCRIPTION_DIR, EVAL_DIR, MIDI_REF_DIR = _resolve_dirs(args.dataset)

    TRANSCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # source.yaml is the annotated reference config, not a runnable pipeline config.
    configs = sorted(c for c in CONFIGS_DIR.glob("*.yaml") if c.stem != "source")
    if not configs:
        print(f"No configs found in {CONFIGS_DIR}", file=sys.stderr)
        return 1

    if args.config is not None:
        available = {c.stem for c in configs}
        unknown = [n for n in args.config if n not in available]
        if unknown:
            print(
                f"error: unknown config(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(available))}",
                file=sys.stderr,
            )
            return 1
        configs = [c for c in configs if c.stem in set(args.config)]

    failures: list[str] = []
    summary_rows: list[dict] = []
    all_rows: list[dict] = []

    def _merge(result: tuple[list[dict], list[dict], list[str]]) -> None:
        s_rows, a_rows, f_list = result
        summary_rows.extend(s_rows)
        all_rows.extend(a_rows)
        failures.extend(f_list)

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    _process_config,
                    config_path,
                    args,
                    AUDIO_DIR,
                    TRANSCRIPTION_DIR,
                    EVAL_DIR,
                    MIDI_REF_DIR,
                )
                for config_path in configs
            ]
            for future in as_completed(futures):
                _merge(future.result())
        summary_rows.sort(key=lambda r: r.get("config", ""))
    else:
        for config_path in configs:
            _merge(
                _process_config(
                    config_path,
                    args,
                    AUDIO_DIR,
                    TRANSCRIPTION_DIR,
                    EVAL_DIR,
                    MIDI_REF_DIR,
                )
            )

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_path = EVAL_DIR / "summary.jsonl"
    _dump_jsonl(summary_rows, summary_path)

    summary_csv_path = EVAL_DIR / "summary.csv"
    _dump_csv(summary_rows, summary_csv_path)

    all_results_csv_path = EVAL_DIR / "all_results.csv"
    _dump_csv(all_rows, all_results_csv_path)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(summary_rows)} configs processed, {len(failures)} failed.")
    if failures:
        print("Failed steps:")
        for f in failures:
            print(f"  {f}")
    print(f"Summary    -> {summary_path}")
    print(f"Summary    -> {summary_csv_path}")
    print(f"All results-> {all_results_csv_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
