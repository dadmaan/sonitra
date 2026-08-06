#!/usr/bin/env python3
"""Download AMT benchmark datasets into the Sonitra corpus directory.

This script is intentionally self-contained — it uses only Python stdlib so it
can be run before the project environment is set up.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py maestro-v3
    python scripts/download_datasets.py bsed
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py maestro-v3 --output-dir /data/corpus
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

REPO: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# To add a new dataset: copy one entry below and fill in the fields.
# `extract_map` is a list of (zip_source_prefix, target_subdir) pairs: any zip
# member whose path starts with a given prefix is extracted (with that prefix
# stripped) under corpus_subdir/target_subdir/. Members matching no prefix are
# skipped. Most datasets need a single ("<top-level-dir>/", "midi") pair;
# multi-modal datasets (e.g. MIDI + audio) can route different zip subfolders
# to different target subdirs.
DATASETS: Dict[str, Dict] = {
    "maestro-v3": {
        "name": "MAESTRO V3.0.0 (MIDI only)",
        "description": (
            "1,276 piano MIDI files paired with professional recordings. "
            "Standard AMT benchmark (Hawthorne et al., ICLR 2019)."
        ),
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "corpus_subdir": "maestro-v3",
        "extract_map": [("maestro-v3.0.0/", "midi")],
    },
    "bsed": {
        "name": "Beethoven Symphony Excerpt Dataset (BSED) v1.0",
        "description": (
            "20 Beethoven symphony excerpts: MIDI scores paired with 4 real concert "
            "recordings + 1 synthetic rendition each (pitch-corrected to A440). "
            "Note-level score-audio alignment annotations and MusicXML/Sibelius scores "
            "are also on Zenodo but not fetched by this script — see ROADMAP.md. "
            "CC BY-NC-SA 4.0 (noncommercial). Berendes et al., TISMIR 2026."
        ),
        "url": "https://zenodo.org/records/20344500/files/BSED.zip",
        "corpus_subdir": "bsed",
        "extract_map": [
            ("BSED_1.0/01_ScoreData/MIDI/", "midi"),
            ("BSED_1.0/02_Audio/wav_44100_440Hz/", "recordings"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_dirs(output_dir: Path, spec: Dict) -> List[Path]:
    """Return one path per distinct target subdir named in spec's extract_map."""
    dataset_dir = output_dir / spec["corpus_subdir"]
    subdirs = sorted({target_subdir for _, target_subdir in spec["extract_map"]})
    return [dataset_dir / subdir for subdir in subdirs]


def _is_already_present(output_dir: Path, spec: Dict) -> bool:
    """Return True when every target subdir exists and contains at least one file."""
    dirs = _target_dirs(output_dir, spec)
    return all(d.is_dir() and any(d.iterdir()) for d in dirs)


def _print_list(output_dir: Path) -> None:
    col_name = max(len(k) for k in DATASETS) + 2
    print(f"{'Dataset':<{col_name}}  {'Target path':<40}  Description")
    print("-" * 120)
    for key, spec in DATASETS.items():
        target = output_dir / spec["corpus_subdir"]
        print(f"{key:<{col_name}}  {str(target):<40}  {spec['description']}")


def _make_reporthook(name: str):
    """Return a urlretrieve reporthook that prints download progress in-place."""

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded: float = block_num * block_size
        if total_size > 0:
            total_mb: float = total_size / 1_048_576
            done_mb: float = min(downloaded, total_size) / 1_048_576
            print(
                f"\rDownloading {name}: {done_mb:.1f} MB / {total_mb:.1f} MB",
                end="",
                flush=True,
            )
        else:
            done_mb = downloaded / 1_048_576
            print(
                f"\rDownloading {name}: {done_mb:.1f} MB",
                end="",
                flush=True,
            )

    return reporthook


def _download_and_extract(key: str, spec: Dict, output_dir: Path) -> int:
    """Download and extract one dataset. Returns the number of files extracted."""
    name: str = spec["name"]
    dataset_dir: Path = output_dir / spec["corpus_subdir"]
    extract_map: List[Tuple[str, str]] = spec["extract_map"]

    tmp_path: str = ""
    tmp_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path = tmp_fd.name
    tmp_fd.close()

    try:
        urlretrieve(spec["url"], tmp_path, reporthook=_make_reporthook(name))
        print()  # newline after the progress line

        files_extracted: int = 0
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                for src_prefix, target_subdir in extract_map:
                    if not member.filename.startswith(src_prefix):
                        continue
                    relative: str = member.filename[len(src_prefix) :]
                    if not relative:
                        break
                    dest: Path = dataset_dir / target_subdir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    files_extracted += 1
                    break

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return files_extracted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AMT benchmark datasets into the Sonitra corpus directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Examples:",
                "  python scripts/download_datasets.py --list",
                "  python scripts/download_datasets.py maestro-v3",
                "  python scripts/download_datasets.py bsed",
                "  python scripts/download_datasets.py --all",
            ]
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "dataset",
        nargs="?",
        metavar="DATASET",
        help="Name of dataset to download (see --list for available names).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available datasets and exit.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Root corpus directory (default: <repo>/corpus). "
            "Dataset files are placed under OUTPUT_DIR/<corpus_subdir>/midi/."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir: Path = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else REPO / "corpus"
    )

    if args.list:
        _print_list(output_dir)
        return 0

    if args.dataset is None and not args.all:
        print(
            "error: specify a dataset name or pass --all. "
            "Use --list to see available datasets.",
            file=sys.stderr,
        )
        return 1

    if args.dataset is not None and args.dataset not in DATASETS:
        available: str = ", ".join(DATASETS.keys())
        print(
            f"error: unknown dataset '{args.dataset}'. Available: {available}",
            file=sys.stderr,
        )
        return 1

    targets: Dict[str, Dict] = (
        DATASETS if args.all else {args.dataset: DATASETS[args.dataset]}
    )

    any_failure: bool = False
    for key, spec in targets.items():
        dataset_dir: Path = output_dir / spec["corpus_subdir"]
        if _is_already_present(output_dir, spec):
            print(f"[skip] {spec['name']} — already present at {dataset_dir}")
            continue

        try:
            n_files: int = _download_and_extract(key, spec, output_dir)
            print(f"[done] {spec['name']}  {n_files} files extracted -> {dataset_dir}")
        except Exception as exc:
            print(f"[error] {spec['name']}: {exc}", file=sys.stderr)
            any_failure = True

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
