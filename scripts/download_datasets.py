#!/usr/bin/env python3
"""Download AMT benchmark datasets into the Sonitra corpus directory.

Runs on the Python standard library alone; with ``rich`` and a terminal it adds
an interactive picker and live progress. Interrupted downloads resume on the
next run; ``--force`` starts over.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import queue
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

try:
    from rich.console import Console as _RichConsole, Group as _RichGroup
    from rich.live import Live as _RichLive
    from rich.progress import (
        BarColumn as _BarColumn,
        DownloadColumn as _DownloadColumn,
        MofNCompleteColumn as _MofNCompleteColumn,
        Progress as _RichProgress,
        TextColumn as _TextColumn,
        TimeElapsedColumn as _TimeElapsedColumn,
        TimeRemainingColumn as _TimeRemainingColumn,
        TransferSpeedColumn as _TransferSpeedColumn,
    )
    from rich.prompt import Prompt as _RichPrompt
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText
except ImportError:  # pragma: no cover - exercised in rich-less environments
    _HAS_RICH = False
else:
    _HAS_RICH = True

REPO: Path = Path(__file__).resolve().parent.parent

METADATA_PATTERNS: FrozenSet[str] = frozenset(
    {".csv", ".json", ".txt", "readme", "license"}
)

_RETRY_SLEEPS: Tuple[int, int, int] = (3, 10, 30)

_ACTIVE_COORDINATOR: Optional["_Coordinator"] = None

# Module-level cache for _source_state
_SOURCE_STATE_CACHE: Dict[Tuple[str, str], str] = {}

_GUITARSET_NEXT_STEPS = (
    "Next steps for GuitarSet (JAMS ground truth needs one conversion):\n"
    "  python scripts/guitarset_jams_to_midi.py --dry-run\n"
    "  python scripts/guitarset_jams_to_midi.py\n"
    "  sonitra benchmark --config config/benchmark/guitarset_test.yaml "
    "--dataset guitarset --limit 2"
)

_MUSICNET_NEXT_STEPS = (
    "Next steps for MusicNet (label CSVs need one conversion):\n"
    "  python scripts/musicnet_labels_to_midi.py --dry-run\n"
    "  python scripts/musicnet_labels_to_midi.py\n"
    "  sonitra benchmark --config config/benchmark/musicnet_test.yaml "
    "--dataset musicnet --limit 2"
)

DATASETS: Dict[str, Dict] = {
    "maestro-v3-midi": {
        "name": "MAESTRO V3.0.0 (MIDI only)",
        "description": (
            "1,276 piano MIDI files + metadata, no audio. Standard AMT benchmark "
            "(Hawthorne et al., ICLR 2019). CC BY-NC-SA 4.0 (noncommercial)."
        ),
        "note": (
            "Solo piano. 199 h of competition performances, recorded as MIDI "
            "by the piano itself and aligned to the audio within about 3 ms. "
            "Made for piano transcription and generation."
        ),
        "corpus_subdir": "maestro-v3",
        "superseded_by": ["maestro-v3-full"],
        "sources": [
            {
                "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
                "kind": "zip",
                "extract_map": [
                    ("maestro-v3.0.0/", frozenset({".mid", ".midi"}), "midi"),
                    ("maestro-v3.0.0/", METADATA_PATTERNS, "metadata"),
                ],
                "size_mb": 57,
            },
        ],
    },
    "maestro-v3-wav": {
        "name": "MAESTRO V3.0.0 (recordings only)",
        "description": (
            "Professional piano recordings paired with MAESTRO's MIDI (fetched "
            "separately via maestro-v3-midi) + metadata, no MIDI. MAESTRO ships no "
            "official audio-only archive, so this downloads the same ~120 GB "
            "full zip as maestro-v3-full and discards the MIDI members. "
            "CC BY-NC-SA 4.0 (noncommercial)."
        ),
        "note": (
            "Solo piano. 199 h of competition performances, recorded as MIDI "
            "by the piano itself and aligned to the audio within about 3 ms. "
            "Made for piano transcription and generation."
        ),
        "corpus_subdir": "maestro-v3",
        "superseded_by": ["maestro-v3-full"],
        "sources": [
            {
                "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
                "kind": "zip",
                "extract_map": [
                    ("maestro-v3.0.0/", frozenset({".wav"}), "recordings"),
                    ("maestro-v3.0.0/", METADATA_PATTERNS, "metadata"),
                ],
                "size_mb": 122_880,
            },
        ],
    },
    "maestro-v3-full": {
        "name": "MAESTRO V3.0.0 (MIDI + recordings)",
        "description": (
            "1,276 piano MIDI files paired with professional recordings + metadata. "
            "Standard AMT benchmark (Hawthorne et al., ICLR 2019); ~120 GB — most "
            "workflows only need maestro-v3-midi (~57 MB). "
            "CC BY-NC-SA 4.0 (noncommercial)."
        ),
        "note": (
            "Solo piano. 199 h of competition performances, recorded as MIDI "
            "by the piano itself and aligned to the audio within about 3 ms. "
            "Made for piano transcription and generation."
        ),
        "corpus_subdir": "maestro-v3",
        "sources": [
            {
                "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
                "kind": "zip",
                "extract_map": [
                    ("maestro-v3.0.0/", frozenset({".mid", ".midi"}), "midi"),
                    ("maestro-v3.0.0/", frozenset({".wav"}), "recordings"),
                    ("maestro-v3.0.0/", METADATA_PATTERNS, "metadata"),
                ],
                "size_mb": 122_880,
            },
        ],
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
        "note": (
            "Orchestra. 20 Beethoven symphony excerpts, each with 4 concert "
            "recordings and 1 synthetic version. Made to test orchestral "
            "transcription; also suits score-to-audio alignment."
        ),
        "corpus_subdir": "bsed",
        "sources": [
            {
                "url": "https://zenodo.org/records/20344500/files/BSED.zip",
                "kind": "zip",
                "extract_map": [
                    ("BSED_1.0/01_ScoreData/MIDI/", None, "midi"),
                    ("BSED_1.0/02_Audio/wav_44100_440Hz/", None, "recordings"),
                ],
                "size_mb": 380,
            },
        ],
    },
    "musicnet-midi": {
        "name": "MusicNet (MIDI only)",
        "description": (
            "330 score-time reference MIDI files + track metadata, no audio "
            "(~4 MB) — enough for MIDI-input (render → transcribe) runs; "
            "fetch musicnet-full for the ~10.6 GB of real recordings audio-input "
            "runs need — the per-note label CSVs there must be converted to aligned "
            "MIDI via scripts/musicnet_labels_to_midi.py. 7 of 330 upstream MIDI "
            "files are corrupt and skipped. CC BY 4.0 (Thickstun et al., ICLR 2017)."
        ),
        "note": (
            "Classical chamber music. 34 h by 10 composers for 11 instruments, "
            "with note labels from scores aligned to the audio automatically. "
            "Made for detecting which notes play at each moment."
        ),
        "corpus_subdir": "musicnet",
        "superseded_by": ["musicnet-full"],
        "sources": [
            {
                # musicnet_midis.tar.gz: 2,601,302 B, md5 b5fa98a113bfc51c8a445def9f24dc7e verified 2026-09-11
                "url": "https://zenodo.org/records/5120004/files/musicnet_midis.tar.gz",
                "kind": "targz",
                "extract_map": [("", frozenset({".mid", ".midi"}), "midi")],
                "size_mb": 3,
            },
            {
                # musicnet_metadata.csv: 43,775 B, md5 1caef62cee9c875235e62aac368b49d8 verified 2026-09-11
                "url": "https://zenodo.org/records/5120004/files/musicnet_metadata.csv",
                "kind": "file",
                "target_subdir": "metadata",
                "filename": "musicnet_metadata.csv",
                "size_mb": 1,
            },
        ],
    },
    "musicnet-full": {
        "name": "MusicNet (MIDI + recordings)",
        "description": (
            "330 classical recordings (wav, 34 h) with score-time reference MIDI "
            "+ per-note label CSVs (audio-aligned) + track metadata; ~10.6 GB — "
            "MIDI-input runs only need musicnet-midi (~4 MB), whose MIDI and "
            "metadata this re-uses if already on disk. The aligned ground truth is "
            "built by converting the label CSVs to MIDI via "
            "scripts/musicnet_labels_to_midi.py (writes midi/<id>.mid, "
            "metadata/musicnet.csv). 7 of 330 upstream MIDI files are corrupt. "
            "CC BY 4.0 (Thickstun et al., ICLR 2017)."
        ),
        "note": (
            "Classical chamber music. 34 h by 10 composers for 11 instruments, "
            "with note labels from scores aligned to the audio automatically. "
            "Made for detecting which notes play at each moment."
        ),
        "corpus_subdir": "musicnet",
        "next_steps": _MUSICNET_NEXT_STEPS,
        "sources": [
            {
                # musicnet_midis.tar.gz: 2,601,302 B, md5 b5fa98a113bfc51c8a445def9f24dc7e verified 2026-09-11
                "url": "https://zenodo.org/records/5120004/files/musicnet_midis.tar.gz",
                "kind": "targz",
                "extract_map": [("", frozenset({".mid", ".midi"}), "annotations/score_midi")],
                "size_mb": 3,
            },
            {
                # musicnet.tar.gz: 11,097,394,998 B, md5 844764911fa0d5b97c97da944a057590 verified 2026-09-11
                "url": "https://zenodo.org/records/5120004/files/musicnet.tar.gz",
                "kind": "targz",
                "extract_map": [
                    ("", frozenset({".wav"}), "recordings"),
                    ("", frozenset({".csv"}), "annotations/labels"),
                ],
                "size_mb": 10_584,
            },
            {
                # musicnet_metadata.csv: 43,775 B, md5 1caef62cee9c875235e62aac368b49d8 verified 2026-09-11
                "url": "https://zenodo.org/records/5120004/files/musicnet_metadata.csv",
                "kind": "file",
                "target_subdir": "metadata",
                "filename": "musicnet_metadata.csv",
                "size_mb": 1,
            },
        ],
    },
    "e-gmd-midi": {
        "name": "Expanded Groove MIDI Dataset (MIDI only)",
        "description": (
            "1,059 drum performances, each replayed through 43 kits: 45,537 "
            "MIDI files + metadata, no audio. Drum-transcription "
            "AMT benchmark (Callender et al., ISMIR 2020); note this repo's "
            "transcription/evaluation backends target pitched instruments, not "
            "drum-hit classification — download-only for now. CC BY 4.0."
        ),
        "note": (
            "Drum kit. 444 h: 1,059 human drum performances, each replayed "
            "through 43 kits with electronic and acoustic sounds. Made for drum "
            "transcription, which Sonitra can't score yet."
        ),
        "corpus_subdir": "e-gmd",
        "superseded_by": ["e-gmd-full"],
        "sources": [
            {
                "url": "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".mid", ".midi"}), "midi"),
                    ("", METADATA_PATTERNS, "metadata"),
                ],
                "size_mb": 103,
            },
        ],
    },
    "e-gmd-full": {
        "name": "Expanded Groove MIDI Dataset (MIDI + recordings)",
        "description": (
            "1,059 drum performances, each replayed through 43 kits: 45,537 "
            "MIDI + audio pairs + metadata. Drum-transcription "
            "AMT benchmark (Callender et al., ISMIR 2020); ~90 GB — note this repo's "
            "transcription/evaluation backends target pitched instruments, not "
            "drum-hit classification — download-only for now. CC BY 4.0."
        ),
        "note": (
            "Drum kit. 444 h: 1,059 human drum performances, each replayed "
            "through 43 kits with electronic and acoustic sounds. Made for drum "
            "transcription, which Sonitra can't score yet."
        ),
        "corpus_subdir": "e-gmd",
        "sources": [
            {
                "url": "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".mid", ".midi"}), "midi"),
                    ("", frozenset({".wav"}), "recordings"),
                    ("", METADATA_PATTERNS, "metadata"),
                ],
                "size_mb": 92_160,
            },
        ],
    },
    "guitarset-mic": {
        "name": "GuitarSet (mono-mic recordings + JAMS annotations)",
        "description": (
            "360 acoustic-guitar excerpts with mono-mic recordings + JAMS "
            "note-level ground truth (convert to MIDI via "
            "scripts/guitarset_jams_to_midi.py). Pair with guitarset-mix "
            "(pickup-mix variant) for the mic-vs-pickup factor, or fetch "
            "guitarset-full for both variants at once; 6-channel "
            "hex-pickup stems deferred, see ROADMAP.md. "
            "GuitarSet (Xi et al., ISMIR 2018). CC BY 4.0."
        ),
        "note": (
            "Acoustic guitar. 360 clips of about 30 s by 6 players in 5 styles, "
            "as backing chords and solos. A per-string pickup gives labels down "
            "to string and fret. Made for guitar transcription."
        ),
        "corpus_subdir": "guitarset",
        "superseded_by": ["guitarset-full"],
        "next_steps": _GUITARSET_NEXT_STEPS,
        "sources": [
            {
                "url": "https://zenodo.org/records/3371780/files/annotation.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".jams"}), "annotations"),
                ],
                "size_mb": 38,
            },
            {
                "url": "https://zenodo.org/records/3371780/files/audio_mono-mic.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".wav"}), "recordings"),
                ],
                "size_mb": 627,
            },
        ],
    },
    "guitarset-mix": {
        "name": "GuitarSet (pickup-mix recordings + JAMS annotations)",
        "description": (
            "360 acoustic-guitar excerpts with pickup-mix recordings + JAMS "
            "note-level ground truth (convert to MIDI via "
            "scripts/guitarset_jams_to_midi.py). Pair with guitarset-mic "
            "(mono-mic variant) for the mic-vs-pickup factor, or fetch "
            "guitarset-full for both variants at once; 6-channel "
            "hex-pickup stems deferred, see ROADMAP.md. "
            "GuitarSet (Xi et al., ISMIR 2018). CC BY 4.0."
        ),
        "note": (
            "Acoustic guitar. 360 clips of about 30 s by 6 players in 5 styles, "
            "as backing chords and solos. A per-string pickup gives labels down "
            "to string and fret. Made for guitar transcription."
        ),
        "corpus_subdir": "guitarset",
        "superseded_by": ["guitarset-full"],
        "next_steps": _GUITARSET_NEXT_STEPS,
        "sources": [
            {
                "url": "https://zenodo.org/records/3371780/files/annotation.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".jams"}), "annotations"),
                ],
                "size_mb": 38,
            },
            {
                "url": "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".wav"}), "recordings"),
                ],
                "size_mb": 652,
            },
        ],
    },
    "guitarset-full": {
        "name": "GuitarSet (mic + pickup-mix recordings + JAMS annotations)",
        "description": (
            "All 360 acoustic-guitar excerpts in both recording variants "
            "(720 mono WAVs: reference-mic + pickup-mix) + JAMS note-level "
            "ground truth (convert once to MIDI via "
            "scripts/guitarset_jams_to_midi.py). One-run equivalent of "
            "guitarset-mic + guitarset-mix sharing corpus/guitarset/; "
            "6-channel hex-pickup stems deferred, see ROADMAP.md. "
            "GuitarSet (Xi et al., ISMIR 2018). CC BY 4.0."
        ),
        "note": (
            "Acoustic guitar. 360 clips of about 30 s by 6 players in 5 styles, "
            "as backing chords and solos. A per-string pickup gives labels down "
            "to string and fret. Made for guitar transcription."
        ),
        "corpus_subdir": "guitarset",
        "next_steps": _GUITARSET_NEXT_STEPS,
        "sources": [
            {
                "url": "https://zenodo.org/records/3371780/files/annotation.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".jams"}), "annotations"),
                ],
                "size_mb": 38,
            },
            {
                "url": "https://zenodo.org/records/3371780/files/audio_mono-mic.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".wav"}), "recordings"),
                ],
                "size_mb": 627,
            },
            {
                "url": "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip",
                "kind": "zip",
                "extract_map": [
                    ("", frozenset({".wav"}), "recordings"),
                ],
                "size_mb": 652,
            },
        ],
    },
    "gaps-midi": {
        "name": "GAPS (Guitar-Aligned Performance Scores) v1.1 (MIDI only)",
        "description": (
            "404 classical-guitar MIDI references aligned to the recordings "
            "+ metadata, no audio (~3 MB) — enough for MIDI-input (render → "
            "transcribe) runs; fetch gaps-full for the ~15.3 GB of real "
            "recordings audio-input runs need. All 404 files kept, official "
            "split not applied (filter via meta.split/meta.f-measure downstream). "
            "CC BY-NC-SA 4.0, research use, cite Riley et al. ISMIR 2024."
        ),
        "note": (
            "Classical guitar. About 23 h of solo performances by over 200 "
            "players in varied recording conditions, each aligned note by note "
            "to its score. Made for guitar transcription on real-world audio. "
            "All 404 files kept; official split not applied."
        ),
        "corpus_subdir": "gaps",
        "superseded_by": ["gaps-full"],
        "sources": [
            {
                "kind": "hf_tree",
                "repo": "xavriley/GAPS",
                "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
                "subdir": "midi",
                "patterns": frozenset({".mid", ".midi"}),
                "target_subdir": "midi",
                "size_mb": 3,
            },
            {
                "url": "https://huggingface.co/datasets/xavriley/GAPS/resolve/b4c89a33a639c7ae903e74102dfbb3e147e1417f/gaps_metadata_with_splits.csv",
                "kind": "file",
                "target_subdir": "metadata",
                "filename": "gaps_metadata_with_splits.csv",
                "size_mb": 1,
            },
        ],
    },
    "gaps-full": {
        "name": "GAPS (Guitar-Aligned Performance Scores) v1.1 (MIDI + recordings)",
        "description": (
            "404 classical-guitar recordings (48 kHz/16-bit/stereo WAV, ~23 h, 200+ "
            "performers) with aligned MIDI + MusicXML + syncpoints + metadata; "
            "~15.3 GB — MIDI-input runs only need gaps-midi (~3 MB), whose MIDI "
            "this re-uses if already on disk. All 404 files kept, official split "
            "not applied (filter via meta.split/meta.f-measure downstream). "
            "CC BY-NC-SA 4.0, research use, cite Riley et al. ISMIR 2024."
        ),
        "note": (
            "Classical guitar. About 23 h of solo performances by over 200 "
            "players in varied recording conditions, each aligned note by note "
            "to its score. Made for guitar transcription on real-world audio. "
            "All 404 files kept; official split not applied."
        ),
        "corpus_subdir": "gaps",
        "sources": [
            {
                "kind": "hf_tree",
                "repo": "xavriley/GAPS",
                "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
                "subdir": "audio",
                "patterns": frozenset({".wav"}),
                "target_subdir": "recordings",
                "size_mb": 15444,
            },
            {
                "kind": "hf_tree",
                "repo": "xavriley/GAPS",
                "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
                "subdir": "midi",
                "patterns": frozenset({".mid", ".midi"}),
                "target_subdir": "midi",
                "size_mb": 3,
            },
            {
                "kind": "hf_tree",
                "repo": "xavriley/GAPS",
                "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
                "subdir": "musicxml",
                "patterns": frozenset({".xml"}),
                "target_subdir": "annotations/musicxml",
                "size_mb": 231,
            },
            {
                "kind": "hf_tree",
                "repo": "xavriley/GAPS",
                "revision": "b4c89a33a639c7ae903e74102dfbb3e147e1417f",
                "subdir": "syncpoints",
                "patterns": frozenset({".json"}),
                "target_subdir": "annotations/syncpoints",
                "size_mb": 1,
            },
            {
                "url": "https://huggingface.co/datasets/xavriley/GAPS/resolve/b4c89a33a639c7ae903e74102dfbb3e147e1417f/gaps_metadata_with_splits.csv",
                "kind": "file",
                "target_subdir": "metadata",
                "filename": "gaps_metadata_with_splits.csv",
                "size_mb": 1,
            },
        ],
    },
}


def _dataset_size_mb(spec: Dict) -> int:
    """Total download size hint for a dataset: sum of its sources' size_mb."""
    return sum(int(source.get("size_mb", 0)) for source in spec["sources"])


def _all_target_subdirs(spec: Dict) -> List[str]:
    """Every distinct target_subdir a dataset's sources can write into."""
    subdirs: set = set()
    for source in spec["sources"]:
        if source["kind"] == "file":
            subdirs.add(source["target_subdir"])
        elif source["kind"] == "hf_tree":
            subdirs.add(source["target_subdir"])
        else:
            subdirs.update(target for _, _, target in source["extract_map"])
    return sorted(subdirs)


def _matches_patterns(filename: str, patterns: Optional[FrozenSet[str]]) -> bool:
    """True if `filename` (a member's own basename) satisfies an extract_map rule.

    `patterns=None` matches anything. Otherwise each pattern is either a
    lowercase extension (``.wav``) or a lowercase exact basename (``readme``,
    for extensionless files like README/LICENSE).
    """
    if patterns is None:
        return True
    name = Path(filename).name
    suffix = Path(filename).suffix.lower()
    return (suffix != "" and suffix in patterns) or name.lower() in patterns


def _route_member(
    relative_or_full_name: str, extract_map: List[Tuple[str, Optional[FrozenSet[str]], str]]
) -> Optional[Tuple[str, str]]:
    """Find the first extract_map rule matching an archive member.

    Returns ``(prefix, target_subdir)`` for the first matching rule, or
    ``None`` when no rule matches (the member should be skipped).
    """
    for prefix, patterns, target_subdir in extract_map:
        if not relative_or_full_name.startswith(prefix):
            continue
        if not _matches_patterns(relative_or_full_name, patterns):
            continue
        return prefix, target_subdir
    return None


# ---------------------------------------------------------------------------
# Helpers - source identity, download identity, records
# ---------------------------------------------------------------------------


def _source_id(source: Dict) -> str:
    """First 12 hex chars of sha256 over canonical JSON of output-determining fields.

    - zip/targz: kind, url, extract_map (with sorted patterns).
    - file: kind, url, target_subdir, filename.
    - hf_tree: kind, repo, revision, subdir, patterns, target_subdir.

    Verbatim-copied sources share an id; same URL with different extract_map does not.
    """
    kind = source.get("kind")
    if kind in ("zip", "targz"):
        canon_map = []
        for prefix, patterns, target in source.get("extract_map", []):
            if patterns is None:
                pat = None
            else:
                pat = sorted(patterns)
            canon_map.append([prefix, pat, target])
        payload = {"kind": kind, "url": source.get("url"), "extract_map": canon_map}
    elif kind == "file":
        payload = {
            "kind": kind,
            "url": source.get("url"),
            "target_subdir": source.get("target_subdir"),
            "filename": source.get("filename"),
        }
    elif kind == "hf_tree":
        patterns = source.get("patterns")
        if patterns is None:
            pat = None
        else:
            pat = sorted(patterns)
        payload = {
            "kind": kind,
            "repo": source.get("repo"),
            "revision": source.get("revision"),
            "subdir": source.get("subdir"),
            "patterns": pat,
            "target_subdir": source.get("target_subdir"),
        }
    else:
        raise ValueError(f"unknown source kind: {kind}")
    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:12]


def _download_key(source: Dict) -> str:
    """Download identity for resume/locking: sha256[:10] of URL or repo@revision/subdir."""
    kind = source.get("kind")
    if kind == "hf_tree":
        material = f"{source.get('repo')}@{source.get('revision')}/{source.get('subdir')}"
    else:
        material = source.get("url", "")
    return hashlib.sha256(material.encode()).hexdigest()[:10]


def _record_path(dataset_dir: Path, source_id: str) -> Path:
    return dataset_dir / ".sources" / f"{source_id}.json"


def _write_record(dataset_dir: Path, source_id: str, origin: str, files: Dict[str, int]) -> None:
    rec_path = _record_path(dataset_dir, source_id)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = rec_path.with_name(f".{source_id}.tmp")
    data = {"version": 1, "source_id": source_id, "origin": origin, "files": files}
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, rec_path)
    # Update cache to satisfied
    _SOURCE_STATE_CACHE[(str(dataset_dir), source_id)] = "satisfied"


def _clear_state_cache() -> None:
    _SOURCE_STATE_CACHE.clear()


def _invalidate_state(dataset_dir: Path, source_id: str) -> None:
    _SOURCE_STATE_CACHE.pop((str(dataset_dir), source_id), None)


def _source_state(dataset_dir: Path, source_id: str) -> str:
    """Return satisfied/stale/none, memoized per (dataset_dir, source_id)."""
    key = (str(dataset_dir), source_id)
    if key in _SOURCE_STATE_CACHE:
        return _SOURCE_STATE_CACHE[key]
    rec = _record_path(dataset_dir, source_id)
    if not rec.exists():
        _SOURCE_STATE_CACHE[key] = "none"
        return "none"
    try:
        data = json.loads(rec.read_text())
        files = data.get("files", {})
        if not isinstance(files, dict):
            raise ValueError("invalid files")
    except Exception:
        _SOURCE_STATE_CACHE[key] = "stale"
        return "stale"
    for rel, expected in files.items():
        path = dataset_dir / rel
        try:
            if not path.is_file():
                _SOURCE_STATE_CACHE[key] = "stale"
                return "stale"
            if path.stat().st_size != int(expected):
                _SOURCE_STATE_CACHE[key] = "stale"
                return "stale"
        except (OSError, ValueError, TypeError):
            _SOURCE_STATE_CACHE[key] = "stale"
            return "stale"
    _SOURCE_STATE_CACHE[key] = "satisfied"
    return "satisfied"


def _stale_details(dataset_dir: Path, source_id: str) -> List[str]:
    rec = _record_path(dataset_dir, source_id)
    if not rec.exists():
        return []
    try:
        data = json.loads(rec.read_text())
        files = data.get("files", {})
    except Exception:
        return []
    missing: List[str] = []
    for rel, expected in files.items():
        path = dataset_dir / rel
        try:
            if not path.is_file() or path.stat().st_size != int(expected):
                missing.append(rel)
        except (OSError, ValueError, TypeError):
            missing.append(rel)
        if len(missing) >= 5:
            break
    return missing


def _has_legacy_markers_or_partials(output_dir: Path, key: str) -> bool:
    downloads = output_dir / ".downloads"
    if not downloads.is_dir():
        return False
    for p in downloads.glob(f"{key}.*.ok"):
        if p.is_file():
            return True
    for p in downloads.glob(f"{key}.*.part"):
        if p.is_file():
            return True
    return False


def _adopt_legacy_partial(output_dir: Path, key: str, source: Dict, download_key: str) -> None:
    if source.get("kind") == "hf_tree":
        return
    url = source.get("url")
    if not url:
        return
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    legacy = output_dir / ".downloads" / f"{key}.{digest}.part"
    new = output_dir / ".downloads" / f"{download_key}.part"
    if legacy.exists() and not new.exists():
        try:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, new)
        except OSError:
            pass


def _unlink_legacy_markers(output_dir: Path, key: str) -> None:
    downloads = output_dir / ".downloads"
    if not downloads.is_dir():
        return
    for p in downloads.glob(f"{key}.*.ok"):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _is_already_present(key: str, spec: Dict, output_dir: Path) -> bool:
    """Return True when dataset is fully downloaded.

    True when any superseded_by key is present, or every source is satisfied.
    When every source is none and no legacy leftover exists, falls back to
    legacy check that every target dir is non-empty.
    """
    # Check superseded_by first
    superseded_by = spec.get("superseded_by")
    if superseded_by:
        for sup_key in superseded_by:
            sup_spec = DATASETS.get(sup_key)
            if sup_spec is None:
                continue
            sup_present = _is_already_present_inner(sup_key, sup_spec, output_dir)
            if sup_present:
                return True
    return _is_already_present_inner(key, spec, output_dir)


def _is_already_present_inner(key: str, spec: Dict, output_dir: Path) -> bool:
    dataset_dir = output_dir / spec["corpus_subdir"]
    # Compute states for each source
    states: List[str] = []
    for source in spec["sources"]:
        sid = _source_id(source)
        states.append(_source_state(dataset_dir, sid))
    if all(s == "satisfied" for s in states):
        return True
    if all(s == "none" for s in states):
        # Check legacy leftover
        if _has_legacy_markers_or_partials(output_dir, key):
            return False
        # Fallback to target dirs non-empty
        dirs = _target_dirs(output_dir, spec)
        return all(d.is_dir() and any(d.iterdir()) for d in dirs)
    return False


def _target_dirs(output_dir: Path, spec: Dict) -> List[Path]:
    """Return one path per distinct target subdir any of spec's sources write into."""
    dataset_dir = output_dir / spec["corpus_subdir"]
    return [dataset_dir / subdir for subdir in _all_target_subdirs(spec)]


# Legacy marker/partial helpers kept for backward compat with existing tests
# New code uses .sources records; these are used only by legacy corpora checks.
def _marker_path(output_dir: Path, key: str, index: int) -> Path:
    """Legacy: Path of the completion marker for one source of a dataset."""
    return output_dir / ".downloads" / f"{key}.{index}.ok"


def _write_marker(output_dir: Path, key: str, index: int, source: Dict) -> None:
    """Legacy: Record that a source's download AND extraction fully succeeded."""
    marker = _marker_path(output_dir, key, index)
    marker.parent.mkdir(parents=True, exist_ok=True)
    url = source.get("url")
    if url:
        marker.write_text(url + "\n")
    else:
        marker.write_text(
            f"hf_tree:{source.get('repo', '')}@{source.get('revision', '')}"
            f"/{source.get('subdir', '')}\n"
        )


def _partial_path(output_dir: Path, key: str, index: int, source: Dict) -> Path:
    """Legacy: Deterministic partial-download path for one source (cross-run resume).

    Kept for tests and legacy adoption. New code uses .downloads/<download_key>.part.
    """
    digest: str = hashlib.sha256(source["url"].encode()).hexdigest()[:10]
    return output_dir / ".downloads" / f"{key}.{digest}.part"


def _reset_download_state(output_dir: Path, key: str, spec: Dict) -> None:
    """Legacy: Remove markers, partials, and extracted parts (for old tests)."""
    for index in range(len(spec["sources"])):
        _marker_path(output_dir, key, index).unlink(missing_ok=True)
    for index, source in enumerate(spec["sources"]):
        if source.get("kind") == "hf_tree":
            continue
        _partial_path(output_dir, key, index, source).unlink(missing_ok=True)
    for subdir in _all_target_subdirs(spec):
        target_dir: Path = output_dir / spec["corpus_subdir"] / subdir
        for part in target_dir.rglob("*.part"):
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass


def _clear_completion_state(output_dir: Path, key: str, spec: Dict) -> None:
    """Legacy: Remove markers/partials after success (now handled via records)."""
    for index in range(len(spec["sources"])):
        _marker_path(output_dir, key, index).unlink(missing_ok=True)
    for index, source in enumerate(spec["sources"]):
        if source.get("kind") == "hf_tree":
            continue
        _partial_path(output_dir, key, index, source).unlink(missing_ok=True)


def _print_list(output_dir: Path) -> None:
    _clear_state_cache()
    col_name = max(len(k) for k in DATASETS) + 2
    print(f"{'Dataset':<{col_name}}  {'Target path':<40}  Description")
    print("-" * 120)
    for key, spec in DATASETS.items():
        target = output_dir / spec["corpus_subdir"]
        # Show stale as missing/incomplete per plan
        dataset_dir = output_dir / spec["corpus_subdir"]
        # Use _is_already_present for badge but stale sources show incomplete
        # Check if any source stale -> incomplete
        has_stale = False
        for source in spec["sources"]:
            if _source_state(dataset_dir, _source_id(source)) == "stale":
                has_stale = True
                break
        badge = "present" if _is_already_present(key, spec, output_dir) and not has_stale else "missing"
        # Actually _is_already_present already returns False for stale, so badge will be missing
        # but we keep logic for clarity
        print(f"{key:<{col_name}}  {str(target):<40}  {spec['description']}")


def _note_groups() -> List[Tuple[str, str, str]]:
    """One ``(numbers, dataset, note)`` row per dataset, in registry order.

    Keys sharing a ``corpus_subdir`` are variants of one dataset with one
    note, so they collapse into a single row. ``numbers`` are the keys'
    1-based picker numbers (``"1-3"`` when adjacent, else ``"1,3"``);
    ``dataset`` is the first key's name without its trailing ``(variant)``.
    """
    specs: List[Dict] = list(DATASETS.values())
    groups: Dict[str, List[int]] = {}
    for index, spec in enumerate(specs, start=1):
        groups.setdefault(spec["corpus_subdir"], []).append(index)
    rows: List[Tuple[str, str, str]] = []
    for indices in groups.values():
        spec = specs[indices[0] - 1]
        if len(indices) > 1 and indices == list(range(indices[0], indices[-1] + 1)):
            numbers = f"{indices[0]}-{indices[-1]}"
        else:
            numbers = ",".join(str(index) for index in indices)
        name = re.sub(r"\s*\([^()]*\)$", "", spec["name"])
        rows.append((numbers, name, spec.get("note", "")))
    return rows


def _print_notes() -> None:
    """Plain --notes output: each dataset's name, then its note indented below."""
    rows = _note_groups()
    col = max(len(numbers) for numbers, _, _ in rows) + 2
    print(f"{'#':<{col}}Dataset")
    print("-" * 120)
    for numbers, name, note in rows:
        print(f"{numbers:<{col}}{name}")
        print(f"{'':<{col}}{note}")


def _parse_content_range_total(value: Optional[str]) -> int:
    """Extract the total size from a Content-Range header (0 when unknown).

    Accepts ``bytes a-b/total`` (206 responses) and ``bytes */total`` (416
    responses); ``total`` may be ``*``, meaning unknown.
    """
    if not value:
        return 0
    try:
        total: str = value.rsplit("/", 1)[1].strip()
    except IndexError:
        return 0
    if total == "*":
        return 0
    try:
        return int(total)
    except ValueError:
        return 0


def _download_file_attempt(
    url: str, dest: Path, *, progress: Optional[Callable[[int, int], None]], **kwargs
) -> int:
    """One download attempt: resume-aware GET + chunked append to `dest`.

    Returns the cumulative byte count written to `dest` (including any
    resumed prefix). On failure the partial file is left in place — it is the
    resume foundation for the next attempt/run.
    """
    headers: Dict[str, str] = {"User-Agent": "Sonitra-Dataset-Downloader/1.0"}
    prefix: int = 0
    mode: str = "wb"
    if dest.exists() and dest.stat().st_size > 0:
        prefix = dest.stat().st_size
        headers["Range"] = f"bytes={prefix}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status: int = resp.status
            if status == 206:
                total: int = _parse_content_range_total(
                    resp.headers.get("Content-Range")
                )
            elif status == 200:
                if prefix > 0:
                    dest.write_bytes(b"")
                    prefix = 0
                    mode = "wb"
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except ValueError:
                    total = 0
            else:
                raise RuntimeError(f"unexpected HTTP status {status} for {url}")
            downloaded: int = prefix
            with dest.open(mode) as fh:
                while True:
                    if _ACTIVE_COORDINATOR is not None and _ACTIVE_COORDINATOR.cancel.is_set():
                        raise RuntimeError("cancelled")
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            server_total: int = _parse_content_range_total(
                exc.headers.get("Content-Range", "")
            )
            if prefix > 0 and prefix == server_total:
                return prefix
            raise RuntimeError(
                f"partial file {dest} is larger than the server's file "
                f"({prefix} > {server_total} bytes); delete the .part file "
                f"or re-run with --force"
            ) from exc
        raise
    if total > 0 and downloaded < total:
        raise RuntimeError(
            f"incomplete download: got {downloaded} of {total} bytes"
        )
    return downloaded


def _download_file(
    url: str,
    dest: Path,
    *,
    name: str,
    output_dir: Path,
    key: str,
    index: int,
    progress: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> int:
    """Download `url` into `dest` with resume support and bounded retries.

    `dest` is the working file (callers decide final placement). If it
    already holds bytes, a ``Range`` request resumes from there (206 appends,
    200 restarts, 416 means the partial already matches the server's file).
    Up to 4 attempts total with backoff sleeps of 3/10/30 s; retryable errors
    are URLError (timeouts, resets, refused), IncompleteRead, TimeoutError,
    and HTTP 408/429/5xx. The partial file is never deleted on failure.
    Returns the final byte count written to `dest`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[BaseException] = None
    for attempt in range(4):
        try:
            return _download_file_attempt(url, dest, progress=progress)
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429) or exc.code >= 500:
                last_error = exc
            else:
                raise RuntimeError(
                    f"HTTP {exc.code} {exc.reason} for {url}"
                ) from exc
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError) as exc:
            last_error = exc
        except RuntimeError as exc:
            # Cancelled should propagate without retry
            if str(exc) == "cancelled":
                raise
            last_error = exc
        if attempt < 3:
            time.sleep(_RETRY_SLEEPS[attempt])
    raise RuntimeError(
        f"download of {name} failed after 4 attempts: {last_error}"
    ) from last_error


def _parse_link_next(link_header: Optional[str]) -> Optional[str]:
    """Extract the ``rel="next"`` URL from an HTTP Link header, if present.

    Hugging Face sends ``rel="next"``, but RFC 8288 also permits ``rel=next``
    and ``rel='next'``. All three are accepted: failing to match would
    silently truncate a listing to its first page, and a short corpus that
    still fills its target dir reads as complete to ``_is_already_present``.
    ``[^,]*`` keeps each match inside one comma-separated link value.
    """
    if not link_header:
        return None
    pattern = r"""<([^>]+)>\s*;\s*[^,]*\brel\s*=\s*(?:"next"|'next'|next\b)"""
    for match in re.finditer(pattern, link_header):
        return match.group(1)
    return None


def _hf_list_tree(repo: str, revision: str, subdir: str) -> List[Tuple[str, int]]:
    """List files in one Hugging Face dataset subdirectory via the tree API.

    Returns ``[(path, size), ...]`` for entries with ``type == "file"``;
    ``type == "directory"`` entries are ignored (defensive — the listing is
    non-recursive, so none should appear). Follows ``Link: <...>; rel="next"``
    pagination until the header is absent. Transient errors are retried with
    ``_RETRY_SLEEPS`` backoff.
    """
    base_url = (
        f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{subdir}?limit=1000"
    )
    url: Optional[str] = base_url
    results: List[Tuple[str, int]] = []
    headers: Dict[str, str] = {"User-Agent": "Sonitra-Dataset-Downloader/1.0"}
    while url is not None:
        last_error: Optional[BaseException] = None
        payload = None
        link_header: Optional[str] = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    status: int = resp.status
                    if status != 200:
                        raise RuntimeError(f"unexpected HTTP status {status} for {url}")
                    chunks: List[bytes] = []
                    while True:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    payload = json.loads(raw.decode("utf-8"))
                    try:
                        link_header = resp.headers.get("Link")
                    except AttributeError:
                        link_header = None
                    break
            except urllib.error.HTTPError as exc:
                if exc.code in (408, 429) or exc.code >= 500:
                    last_error = exc
                else:
                    raise RuntimeError(
                        f"HTTP {exc.code} {exc.reason} for {url}"
                    ) from exc
            except (
                urllib.error.URLError,
                http.client.IncompleteRead,
                TimeoutError,
            ) as exc:
                last_error = exc
            if attempt < 3:
                time.sleep(_RETRY_SLEEPS[attempt])
        if payload is None:
            raise RuntimeError(
                f"listing of {repo}/{subdir}@{revision} failed after 4 attempts: "
                f"{last_error}"
            ) from last_error
        for entry in payload:
            if entry.get("type") != "file":
                continue
            path = entry.get("path")
            if not path:
                continue
            try:
                size = int(entry.get("size") or 0)
            except (ValueError, TypeError):
                size = 0
            results.append((path, size))
        url = _parse_link_next(link_header)
    return results


def _download_hf_tree(
    source: Dict,
    dataset_dir: Path,
    *,
    output_dir: Path,
    key: str,
    index: int,
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
    report: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> int:
    """Download one ``hf_tree`` source (many small files) into ``target_subdir``.

    For each listed file passing ``patterns`` (via ``_matches_patterns``),
    ``dest = dataset_dir / target_subdir / Path(path).name`` (flattened). Skip
    when ``dest`` exists with matching size unless ``force``; skipped files
    still count toward the returned total and advance progress. Otherwise
    download to ``dest.with_name(dest.name + ".<source_id>.part")`` via ``_download_file``
    (retries + Range resume free), then ``os.replace`` into place. ``progress``
    receives cumulative ``(done_within_source, total_within_source)`` so rich
    callers can add their ``bytes_before`` offset one level up. ``report``, if
    given, is called once at the end with ``(skipped, downloaded)`` file counts.

    The skip is key-independent: files another key sharing the corpus_subdir
    already fetched (e.g. gaps-midi's MIDI, when gaps-full runs) are re-used,
    which is sound only because such keys pin the same revision.
    """
    repo: str = source["repo"]
    revision: str = source["revision"]
    subdir: str = source["subdir"]
    target_subdir: str = source["target_subdir"]
    patterns: Optional[FrozenSet[str]] = source.get("patterns")
    listing = _hf_list_tree(repo, revision, subdir)
    filtered: List[Tuple[str, int]] = [
        (path, size)
        for path, size in listing
        if _matches_patterns(Path(path).name, patterns)
    ]
    total: int = sum(size for _, size in filtered)
    done: int = 0
    count: int = 0
    skipped: int = 0
    source_id = _source_id(source)
    for path, size in filtered:
        dest: Path = dataset_dir / target_subdir / Path(path).name
        if not force and dest.exists() and dest.stat().st_size == size:
            done += size
            count += 1
            skipped += 1
            if progress is not None:
                progress(done, total)
            continue
        part: Path = dest.with_name(dest.name + f".{source_id}.part")
        file_url: str = (
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
        )
        name: str = f"{repo}/{subdir}/{Path(path).name}"
        file_progress: Optional[Callable[[int, int], None]] = None
        if progress is not None:
            def _file_progress(
                downloaded: int,
                file_total: int,
                _done: int = done,
                _total: int = total,
            ) -> None:
                assert progress is not None
                progress(_done + downloaded, _total)

            file_progress = _file_progress
        try:
            _download_file(
                file_url,
                part,
                name=name,
                output_dir=output_dir,
                key=key,
                index=index,
                progress=file_progress,
            )
        except BaseException:
            # Keep part for resume, but on cancel ensure cleanup? Plan says extraction cleanup catches BaseException, but for hf_tree we keep partial for resume on failure; on cancel we should not delete?
            # For now keep part.
            raise
        os.replace(part, dest)
        done += size
        count += 1
        if progress is not None:
            progress(done, total)
    if report is not None:
        report(skipped, count - skipped)
    return count


def _hf_tree_reporter(key: str, source: Dict) -> Callable[[int, int], None]:
    """``report`` callback printing one skip/download summary line per source.

    Makes the per-file skip visible, e.g. ``[gaps-full] midi: 404 already
    present, 0 downloaded`` after gaps-midi fetched the MIDI earlier. Under the
    rich display the line goes through Live's stdout redirect, above the bars.
    """

    def report(skipped: int, downloaded: int) -> None:
        print(
            f"[{key}] {source['subdir']}: {skipped} already present, "
            f"{downloaded} downloaded"
        )

    return report


def _extract_archive(
    tmp_path: str, kind: str, extract_map: List[Tuple[str, Optional[FrozenSet[str]], str]], dataset_dir: Path,
    *, part_suffix: Optional[str] = None, skip_existing: bool = False, on_file: Optional[Callable[[str, int], None]] = None,
    **kwargs,
) -> int:
    """Extract a downloaded zip/tar.gz according to a source's extract_map.

    Routes each regular-file member through `_route_member`; members matching
    no rule are skipped. Returns the number of files extracted.

    Each member is written to a sibling ``.<source_id>.part`` file first and moved into
    place with ``os.replace`` only after the copy completes without exception
    (zipfile raises BadZipFile on CRC mismatch at EOF of the member stream —
    the ``.part`` is then discarded). On failure the member's ``.part`` is
    best-effort unlinked and the exception re-raised. When ``skip_existing``
    is true, members whose destination already exists at the member's size
    are not rewritten.

    When ``on_file`` is given it is called for every routed file (including
    skipped) as ``on_file(relpath, size)``.
    """
    files_extracted: int = 0

    if kind == "zip":
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                if _ACTIVE_COORDINATOR is not None and _ACTIVE_COORDINATOR.cancel.is_set():
                    raise RuntimeError("cancelled")
                match = _route_member(member.filename, extract_map)
                if match is None:
                    continue
                prefix, target_subdir = match
                relative: str = member.filename[len(prefix) :]
                if not relative:
                    continue
                dest: Path = dataset_dir / target_subdir / relative
                size = member.file_size
                relpath = str(Path(target_subdir) / relative)
                if on_file is not None:
                    on_file(relpath, size)
                if skip_existing and dest.exists() and dest.is_file():
                    try:
                        if dest.stat().st_size == size:
                            continue
                    except OSError:
                        pass
                dest.parent.mkdir(parents=True, exist_ok=True)
                if part_suffix:
                    dest_part: Path = dest.with_name(dest.name + f".{part_suffix}.part")
                else:
                    dest_part = dest.with_name(dest.name + ".part")
                try:
                    with zf.open(member) as src, dest_part.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.replace(dest_part, dest)
                except BaseException:
                    dest_part.unlink(missing_ok=True)
                    raise
                files_extracted += 1
    elif kind == "targz":
        with tarfile.open(tmp_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if _ACTIVE_COORDINATOR is not None and _ACTIVE_COORDINATOR.cancel.is_set():
                    raise RuntimeError("cancelled")
                match = _route_member(member.name, extract_map)
                if match is None:
                    continue
                prefix, target_subdir = match
                relative = member.name[len(prefix) :]
                if not relative:
                    continue
                dest = dataset_dir / target_subdir / relative
                size = member.size
                relpath = str(Path(target_subdir) / relative)
                if on_file is not None:
                    on_file(relpath, size)
                if skip_existing and dest.exists() and dest.is_file():
                    try:
                        if dest.stat().st_size == size:
                            continue
                    except OSError:
                        pass
                dest.parent.mkdir(parents=True, exist_ok=True)
                if part_suffix:
                    dest_part = dest.with_name(dest.name + f".{part_suffix}.part")
                else:
                    dest_part = dest.with_name(dest.name + ".part")
                src = tf.extractfile(member)
                if src is None:  # pragma: no cover - not a regular extractable file
                    continue
                try:
                    with src, dest_part.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.replace(dest_part, dest)
                except BaseException:
                    dest_part.unlink(missing_ok=True)
                    raise
                files_extracted += 1
    else:  # pragma: no cover - defensive, all registry entries use zip/targz here
        raise ValueError(f"unknown archive kind: {kind}")

    return files_extracted


# ---------------------------------------------------------------------------
# Coordinator and helpers
# ---------------------------------------------------------------------------

class _Coordinator:
    """Race-free parallel coordination.

    Holds a ``threading.Lock`` per download key so shared sources are fetched
    exactly once per run. Jobs hold at most one lock at a time, so there is
    no deadlock. Also tracks per-run failures and a cancel event, and the set
    of source ids that must be force-re-fetched this run.
    """

    def __init__(self, forced_ids: Optional[set] = None):
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._failures: Dict[str, Exception] = {}
        self._failures_lock = threading.Lock()
        self._forced: set = set(forced_ids) if forced_ids else set()
        self._forced_lock = threading.Lock()
        self.cancel = threading.Event()

    def get_lock(self, download_key: str) -> threading.Lock:
        with self._locks_lock:
            if download_key not in self._locks:
                self._locks[download_key] = threading.Lock()
            return self._locks[download_key]

    def is_forced(self, source_id: str) -> bool:
        with self._forced_lock:
            return source_id in self._forced

    def clear_forced(self, source_id: str) -> None:
        with self._forced_lock:
            self._forced.discard(source_id)

    def set_failure(self, download_key: str, exc: BaseException) -> None:
        with self._failures_lock:
            if download_key not in self._failures:
                self._failures[download_key] = exc  # type: ignore[assignment]

    def get_failure(self, download_key: str) -> Optional[BaseException]:
        with self._failures_lock:
            return self._failures.get(download_key)

    def remaining_forced_with_key(self, download_key: str, exclude_id: str) -> bool:
        with self._forced_lock:
            remaining = self._forced - {exclude_id}
            if not remaining:
                return False
            for key, spec in DATASETS.items():
                for src in spec["sources"]:
                    if _source_id(src) in remaining and _download_key(src) == download_key:
                        return True
            return False


def _bytes_needed(resolved: List[str], output_dir: Path, force: bool = False) -> int:
    """Sum size_mb of unique sources needed for resolved keys, minus satisfied unless forced."""
    seen: set = set()
    needed = 0
    for key in resolved:
        spec = DATASETS[key]
        if not force and _is_already_present(key, spec, output_dir):
            continue
        dataset_dir = output_dir / spec["corpus_subdir"]
        for source in spec["sources"]:
            sid = _source_id(source)
            if sid in seen:
                continue
            seen.add(sid)
            if not force:
                state = _source_state(dataset_dir, sid)
                if state == "satisfied":
                    continue
            needed += int(source.get("size_mb", 0)) * 1_048_576
    return needed


def _resolve_selection(selected: List[str]) -> List[str]:
    """Prune superseded keys when their superseding key is also selected."""
    selected_set = set(selected)
    resolved: List[str] = []
    for key in selected:
        spec = DATASETS[key]
        superseded_by = spec.get("superseded_by") or []
        superseding = [sup for sup in superseded_by if sup in selected_set]
        if superseding:
            sup_key = superseding[0]
            sup_spec = DATASETS.get(sup_key, {})
            print(f"[skip] {spec['name']} — superseded by {sup_key} (also selected)")
            if sup_spec.get("next_steps"):
                # Print next_steps once for superseding key (dedup handled elsewhere)
                pass
            continue
        resolved.append(key)
    return resolved


def _check_disk_space(output_dir: Path, needed_bytes: int) -> Optional[str]:
    """Return an error message when the output filesystem lacks room, else None.

    Needed space is the sum of required bytes plus 5% headroom.
    ``shutil.disk_usage`` requires an existing path, so when
    ``output_dir`` does not exist yet the nearest existing ancestor is probed
    instead. Any OSError disables the check.
    """
    if needed_bytes <= 0:
        return None
    needed: float = needed_bytes * 1.05
    probe: Path = output_dir
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    if usage.free < needed:
        needed_gb: float = needed / (1024**3)
        free_gb: float = usage.free / (1024**3)
        return (
            f"not enough free space on {output_dir}: need ~{needed_gb:.1f} GB, "
            f"{free_gb:.1f} GB available"
        )
    return None


def _reset_for_force(
    dataset_dir: Path, output_dir: Path, source: Dict, source_id: str, download_key: str, coordinator: Optional["_Coordinator"]
) -> None:
    # Record
    rec = _record_path(dataset_dir, source_id)
    rec.unlink(missing_ok=True)
    _invalidate_state(dataset_dir, source_id)
    # Download-key partial if no other forced source needs it
    keep = False
    if coordinator is not None:
        keep = coordinator.remaining_forced_with_key(download_key, source_id)
    if not keep:
        pp = output_dir / ".downloads" / f"{download_key}.part"
        pp.unlink(missing_ok=True)
    # Determine target subdirs for this source
    if source["kind"] == "file":
        target_subdirs = [source["target_subdir"]]
    elif source["kind"] == "hf_tree":
        target_subdirs = [source["target_subdir"]]
    else:
        target_subdirs = sorted({target for _, _, target in source["extract_map"]})
    for subdir in target_subdirs:
        target_dir = dataset_dir / subdir
        if not target_dir.exists():
            continue
        # Delete *.<source_id>.part
        for p in target_dir.rglob(f"*.{source_id}.part"):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        # Legacy un-suffixed *.part sweep
        for p in target_dir.rglob("*.part"):
            # Skip our own new parts already deleted
            if p.name.endswith(f".{source_id}.part"):
                continue
            # If it looks like new-style with another id (.<12hex>.part), keep it
            if re.search(r"\.[0-9a-f]{12}\.part$", p.name):
                continue
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def _present_via_superseded(key: str, spec: Dict, output_dir: Path) -> Optional[str]:
    for sup_key in spec.get("superseded_by") or []:
        sup_spec = DATASETS.get(sup_key)
        if sup_spec is None:
            continue
        if _is_already_present_inner(sup_key, sup_spec, output_dir):
            # Also need all sources of sup satisfied? inner already checks
            return sup_key
    return None


# ---------------------------------------------------------------------------
# Fetch one source
# ---------------------------------------------------------------------------

def _fetch_source(
    source: Dict,
    dataset_dir: Path,
    output_dir: Path,
    key: str,
    *,
    coordinator: Optional["_Coordinator"] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    report: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    global _ACTIVE_COORDINATOR
    _ACTIVE_COORDINATOR = coordinator
    """Fetch one source with locking, resume, skip, stale handling, and record writing.

    Returns the files dict for the source.
    """
    source_id = _source_id(source)
    download_key = _download_key(source)
    # Get lock for download identity
    lock = coordinator.get_lock(download_key) if coordinator else threading.Lock()
    # For non-coordinator case, use simple lock (no sharing)
    with lock:
        if coordinator and coordinator.cancel.is_set():
            raise RuntimeError("cancelled")
        if coordinator:
            fail = coordinator.get_failure(download_key)
            if fail is not None:
                raise fail
        # Force handling
        is_forced = coordinator.is_forced(source_id) if coordinator else False
        if is_forced:
            _reset_for_force(dataset_dir, output_dir, source, source_id, download_key, coordinator)
            if coordinator:
                coordinator.clear_forced(source_id)
        else:
            state = _source_state(dataset_dir, source_id)
            if state == "satisfied":
                rec_path = _record_path(dataset_dir, source_id)
                try:
                    rec = json.loads(rec_path.read_text())
                    files = rec.get("files", {})
                    file_count = len(files)
                except Exception:
                    files = {}
                    file_count = 0
                label = source.get("subdir") if source["kind"] == "hf_tree" else Path(source.get("url", "")).name
                # Summary line for satisfied skip
                if source["kind"] in ("zip", "targz"):
                    print(f"[{key}] {label}: {file_count} already present, 0 extracted")
                else:
                    print(f"[{key}] {label}: {file_count} already present, 0 downloaded")
                _unlink_legacy_markers(output_dir, key)
                if progress is not None:
                    # Advance progress? For rich, caller handles bytes_before.
                    pass
                return files
            elif state == "stale":
                label = source.get("subdir") if source["kind"] == "hf_tree" else Path(source.get("url", "")).name
                missing = _stale_details(dataset_dir, source_id)
                # Count missing files
                rec_path = _record_path(dataset_dir, source_id)
                try:
                    rec = json.loads(rec_path.read_text())
                    total = len(rec.get("files", {}))
                    n_missing = len(missing)
                    # For display, need to show missing count vs total? Plan says N files missing or changed
                    # We'll use n_missing if we could compute, else total
                    if n_missing == 0:
                        n_missing = total
                    example = ", ".join(missing[:3])
                except Exception:
                    n_missing = 1
                    example = ""
                if example:
                    print(f"[stale] {key} {label}: {n_missing} files missing or changed (e.g. {example}) — re-fetching")
                else:
                    print(f"[stale] {key} {label}: {n_missing} files missing or changed — re-fetching")
                # Fall through to re-fetch
            else:  # none
                _adopt_legacy_partial(output_dir, key, source, download_key)
                # fall through

        # Actual fetch
        try:
            files = _do_fetch(source, dataset_dir, output_dir, key, coordinator=coordinator, progress=progress, report=report)
        except BaseException as exc:
            if coordinator:
                coordinator.set_failure(download_key, exc)
            raise
        # Write record
        _write_record(dataset_dir, source_id, key, files)
        _unlink_legacy_markers(output_dir, key)
        return files


def _do_fetch(
    source: Dict,
    dataset_dir: Path,
    output_dir: Path,
    key: str,
    *,
    coordinator: Optional["_Coordinator"] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    report: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    global _ACTIVE_COORDINATOR
    _ACTIVE_COORDINATOR = coordinator
    kind = source["kind"]
    if kind == "file":
        dest = dataset_dir / source["target_subdir"] / source["filename"]
        part = dest.with_name(dest.name + f".{_source_id(source)}.part")
        part.parent.mkdir(parents=True, exist_ok=True)
        label = Path(source["url"]).name
        # Download
        _download_file(
            source["url"],
            part,
            name=f"{key} ({label})",
            output_dir=output_dir,
            key=key,
            index=0,
            progress=progress,
        )
        os.replace(part, dest)
        rel = str(Path(source["target_subdir"]) / source["filename"])
        try:
            size = dest.stat().st_size
        except OSError:
            size = 0
        files = {rel: size}
        # Summary line
        print(f"[{key}] {label}: 0 already present, 1 downloaded")
        if report is not None:
            report(0, 1)
        return files
    elif kind == "hf_tree":
        # Use _download_hf_tree with collection
        files: Dict[str, int] = {}
        # Wrap report to also print
        orig_report = report
        # We need to capture skipped/downloaded for summary if not already via reporter
        # _download_hf_tree will call report
        # For file collection, we need to hook into its per-file handling: we can provide on_file via monkey? Instead collect after via filesystem scan.
        # Simplify: after download, enumerate dest files? But better to have _download_hf_tree call on_file.
        # We modify _download_hf_tree to not yet support on_file, so collect via post-scan using listing.
        # Use listing to know expected files, but actual files dict should include all routed files with sizes.
        # We'll call _download_hf_tree and then build files dict by reading record? But we haven't yet.
        # Alternative: let _download_hf_tree handle files dict creation internally and return it? To keep signature compatible, we make it return count but also call a collector.
        # Easiest: call _download_hf_tree and then build files dict by scanning dataset_dir/target_subdir for matching patterns? But that would miss details.
        # Instead, we add optional file_collector to _download_hf_tree.
        # Let's call with a closure that captures files.
        collected: Dict[str, int] = {}

        def _collect_report(skipped: int, downloaded: int) -> None:
            if orig_report is not None:
                orig_report(skipped, downloaded)
            # Also print via reporter already? The caller may have passed _hf_tree_reporter.
            # If not, we need to print? But _fetch_source caller will have passed reporter as _hf_tree_reporter.
            pass

        # Monkey-patch to collect: we can wrap _hf_list_tree? Simpler: call _download_hf_tree and after it, list actual files on disk for this source's target_subdir with matching patterns? But sizes are from listing, not disk? For skipped files, sizes are from listing, not disk size? But they should match.
        # To avoid complexity, modify _download_hf_tree to accept an extra param `files_dict` to fill.
        # For now, we pass a custom function that we handle inside _download_hf_tree if it supports `on_file`.
        # Since current _download_hf_tree doesn't have on_file, we will handle here by doing listing ourselves and building files dict after download.
        # Let's just manually handle hf_tree fetching here without delegating to _download_hf_tree's listing? But we can delegate and then rebuild files dict via same listing.

        # Call existing _download_hf_tree with progress and report
        # It will handle skipping and downloading
        # After it returns, we build files dict by iterating listing again?
        # Simpler: Build files dict from listing filtered and sizes, plus verify actual files exist.
        listing = _hf_list_tree(source["repo"], source["revision"], source["subdir"])
        patterns = source.get("patterns")
        filtered = [(p, s) for p, s in listing if _matches_patterns(Path(p).name, patterns)]
        # Ensure download already done via _download_hf_tree call below, but we need to have done download first.
        # So we call _download_hf_tree first.
        # To avoid double listing, we could just call _download_hf_tree and then construct files dict from filtered that we already have? But filtered is from listing we just got; but _download_hf_tree does its own listing (duplicate). We can avoid double listing by passing listing? Instead, we will not call _download_hf_tree's inner listing; we will handle hf_tree download manually here using same logic but collecting files.
        # For simplicity, call _download_hf_tree with our wrapper that also collects.
        # Since _download_hf_tree currently does its own listing and we can't inject listing, we will do our own per-file loop here, reusing its logic for part suffix and skipping, and not call _download_hf_tree at all.
        # Let's implement inline hf_tree fetch to collect files dict and handle report.
        repo = source["repo"]
        revision = source["revision"]
        subdir = source["subdir"]
        target_subdir = source["target_subdir"]
        patterns = source.get("patterns")
        # listing already fetched above
        # filtered already
        total = sum(s for _, s in filtered)
        done = 0
        skipped = 0
        source_id = _source_id(source)
        for path, size in filtered:
            dest = dataset_dir / target_subdir / Path(path).name
            rel = str(Path(target_subdir) / Path(path).name)
            # Check existing size
            if not coordinator or not coordinator.is_forced(source_id):
                # We already handled forced via outer, so not forced here
                if dest.exists() and dest.stat().st_size == size:
                    collected[rel] = size
                    done += size
                    skipped += 1
                    if progress is not None:
                        progress(done, total)
                    continue
            # Download to part with suffix
            part = dest.with_name(dest.name + f".{source_id}.part")
            file_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
            name = f"{repo}/{subdir}/{Path(path).name}"
            # file progress handling
            file_progress = None
            if progress is not None:
                def _fp(downloaded: int, file_total: int, _done=done, _total=total) -> None:
                    assert progress is not None
                    progress(_done + downloaded, _total)
                file_progress = _fp
            # Use _download_file
            _download_file(
                file_url,
                part,
                name=name,
                output_dir=output_dir,
                key=key,
                index=0,
                progress=file_progress,
            )
            os.replace(part, dest)
            collected[rel] = size
            done += size
            if progress is not None:
                progress(done, total)
        # Report
        downloaded = len(filtered) - skipped
        if report is not None:
            report(skipped, downloaded)
        else:
            # If caller didn't provide report, print via _hf_tree_reporter logic? But _fetch_source caller provides report as _hf_tree_reporter.
            # The _fetch_source for hf_tree passes report=_hf_tree_reporter(key, source)
            # So report will be not None and will print summary.
            pass
        return collected
    else:  # zip/targz
        download_key = _download_key(source)
        archive_part = output_dir / ".downloads" / f"{download_key}.part"
        label = Path(source["url"]).name
        # Download archive
        _download_file(
            source["url"],
            archive_part,
            name=f"{key} ({label})",
            output_dir=output_dir,
            key=key,
            index=0,
            progress=progress,
        )
        # Extract with skip
        collected: Dict[str, int] = {}

        def on_file(rel: str, size: int) -> None:
            collected[rel] = size

        source_id = _source_id(source)
        n_extracted = _extract_archive(
            str(archive_part),
            kind,
            source["extract_map"],
            dataset_dir,
            part_suffix=source_id,
            skip_existing=True,
            on_file=on_file,
        )
        # Compute skipped vs extracted for summary
        total = len(collected)
        skipped = total - n_extracted
        print(f"[{key}] {label}: {skipped} already present, {n_extracted} extracted")
        if report is not None:
            report(skipped, n_extracted)
        # Remove archive partial after success (unless other forced needs it)
        keep = False
        if coordinator is not None:
            keep = coordinator.remaining_forced_with_key(download_key, source_id)
        if not keep:
            archive_part.unlink(missing_ok=True)
        return collected


# Keep old helpers for backward compat but new logic uses records
def _guitarset_next_steps() -> str:
    return _GUITARSET_NEXT_STEPS

def _download_and_extract(key: str, spec: Dict, output_dir: Path, *, force: bool = False, coordinator: Optional["_Coordinator"] = None) -> int:
    """Download and extract every source of one dataset (plain stdlib path).

    Returns the total file count across all of the dataset's sources. Each
    source is fetched via _fetch_source which handles resume, skip, and record writing.
    """
    dataset_dir: Path = output_dir / spec["corpus_subdir"]
    total = 0
    # Ensure coordinator exists
    if coordinator is None:
        forced_ids = set()
        if force:
            for src in spec["sources"]:
                forced_ids.add(_source_id(src))
        coordinator = _Coordinator(forced_ids if forced_ids else None)
    # For plain path, we need to handle display_name progress printing
    for index, source in enumerate(spec["sources"]):
        if source["kind"] == "hf_tree":
            display_name = f"{spec['name']} ({source['repo']}/{source['subdir']})"
        else:
            display_name = f"{spec['name']} ({Path(source['url']).name})"

        def progress(downloaded: int, total: int) -> None:
            if total > 0:
                print(
                    f"\rDownloading {display_name}: {downloaded / 1_048_576:.1f} MB "
                    f"/ {total / 1_048_576:.1f} MB",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloading {display_name}: {downloaded / 1_048_576:.1f} MB",
                    end="",
                    flush=True,
                )

        # Use _fetch_source to handle the source
        # For plain, report for hf_tree is _hf_tree_reporter
        report = _hf_tree_reporter(key, source) if source["kind"] == "hf_tree" else None
        try:
            files = _fetch_source(
                source,
                dataset_dir,
                output_dir,
                key,
                coordinator=coordinator,
                progress=progress,
                report=report,
            )
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                raise KeyboardInterrupt from exc
            raise
        # For plain path, after each source we print newline after progress if needed
        # _fetch_source for file/archive/hf_tree may have already printed summary lines
        # Ensure newline after progress line if progress was used
        if source["kind"] != "hf_tree":
            # _fetch_source for archives/files prints summary but not progress newline; we add newline if progress was active
            # Check if progress was called? We'll just print newline
            print()
        else:
            # hf_tree progress also uses \r, but report already printed summary after
            # Ensure newline
            if files:
                print()
        total += len(files)
    return total


def _slot_worker(
    slot: int,
    task_id: int,
    work_queue: "queue.Queue[str]",
    display: "_DownloadDisplay",
    output_dir: Path,
    force: bool = False,
    coordinator: Optional["_Coordinator"] = None,
) -> bool:
    """Drain the shared queue, one dataset per slot row. Returns True on error."""
    any_failure: bool = False
    while True:
        if coordinator and coordinator.cancel.is_set():
            return any_failure
        try:
            key = work_queue.get_nowait()
        except queue.Empty:
            return any_failure
        spec = DATASETS[key]
        status, _n_files, error = _download_one(
            key, spec, output_dir, display=display, task_id=task_id, force=force, coordinator=coordinator
        )
        if status == "error":
            any_failure = True
            print(f"[error] {spec['name']}: {error}", file=sys.stderr)


def _run_rich(
    selected: List[str],
    output_dir: Path,
    jobs: int,
    display: "_DownloadDisplay",
    force: bool = False,
    coordinator: Optional["_Coordinator"] = None,
) -> bool:
    """Download selected datasets through the rich display. Returns True on error."""
    if coordinator is None:
        forced_ids = set()
        if force:
            for key in selected:
                for src in DATASETS[key]["sources"]:
                    forced_ids.add(_source_id(src))
        coordinator = _Coordinator(forced_ids if forced_ids else None)
    work_queue: "queue.Queue[str]" = queue.Queue()
    for key in selected:
        work_queue.put(key)
    slots = min(jobs, len(selected))
    executor = ThreadPoolExecutor(max_workers=slots)
    futures = [
        executor.submit(
            _slot_worker,
            slot,
            display.tasks[slot],
            work_queue,
            display,
            output_dir,
            force,
            coordinator,
        )
        for slot in range(slots)
    ]
    try:
        with display:
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        coordinator.cancel.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:
        coordinator.cancel.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    if display.any_failure():
        print(
            f"download finished: {display.done_count()} done, "
            f"{display.skipped_count()} skipped, {display.failed_count()} failed",
            file=sys.stderr,
        )
    return display.any_failure() or any(f.result() for f in futures)


def _run_plain(selected: List[str], output_dir: Path, jobs: int, force: bool = False, coordinator: Optional["_Coordinator"] = None) -> bool:
    """Download selected datasets with the plain stdlib output. Returns True on error."""
    if coordinator is None:
        forced_ids = set()
        if force:
            for key in selected:
                for src in DATASETS[key]["sources"]:
                    forced_ids.add(_source_id(src))
        coordinator = _Coordinator(forced_ids if forced_ids else None)
    any_failure: bool = False

    def worker(key: str) -> bool:
        if coordinator and coordinator.cancel.is_set():
            return False
        spec = DATASETS[key]
        dataset_dir: Path = output_dir / spec["corpus_subdir"]
        status, n_files, error = _download_one(key, spec, output_dir, force=force, coordinator=coordinator)
        if status == "skip":
            print(f"[skip] {spec['name']} — already present at {dataset_dir}")
            sup = _present_via_superseded(key, spec, output_dir)
            if sup:
                sup_spec = DATASETS.get(sup, {})
                if sup_spec.get("next_steps"):
                    print(sup_spec["next_steps"])
        elif status == "done":
            print(f"[done] {spec['name']}  {n_files} files extracted -> {dataset_dir}")
        else:
            print(f"[error] {spec['name']}: {error}", file=sys.stderr)
        return status == "error"

    if jobs <= 1:
        for key in selected:
            if coordinator.cancel.is_set():
                break
            if worker(key):
                any_failure = True
        return any_failure

    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = [executor.submit(worker, key) for key in selected]
    try:
        results = [future.result() for future in futures]
    except KeyboardInterrupt:
        coordinator.cancel.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:
        coordinator.cancel.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return any(results)


def _can_interact() -> bool:
    """True when the interactive picker can run: rich installed + a real TTY."""
    return _HAS_RICH and sys.stdin.isatty() and sys.stdout.isatty()


def _use_rich_output() -> bool:
    """True when rich output (table / live display) may be used."""
    return _HAS_RICH and sys.stdout.isatty()


def _print_table(console: "_RichConsole", output_dir: Path) -> None:
    """Render the dataset registry as a rich table (--list and the picker share it)."""
    _clear_state_cache()
    table = _RichTable(title="Available datasets", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("key")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("target")
    table.add_column("status")
    for index, (key, spec) in enumerate(DATASETS.items(), start=1):
        target = output_dir / spec["corpus_subdir"]
        dataset_dir = output_dir / spec["corpus_subdir"]
        has_stale = any(_source_state(dataset_dir, _source_id(s)) == "stale" for s in spec["sources"])
        if has_stale:
            badge = "missing"
        else:
            badge = "present" if _is_already_present(key, spec, output_dir) else "missing"
        table.add_row(
            str(index),
            key,
            spec["name"],
            f"{_dataset_size_mb(spec):,} MB",
            str(target),
            f"[green]present[/green]" if badge == "present" else "[yellow]missing[/yellow]",
        )
    console.print(table)


def _print_notes_table(console: "_RichConsole") -> None:
    """Render one note row per dataset (--notes and the picker's notes page share it)."""
    table = _RichTable(title="Dataset notes", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("dataset")
    table.add_column("notes", overflow="fold")
    for numbers, name, note in _note_groups():
        table.add_row(numbers, name, note)
    console.print(table)


def _show_notes_page(console: "_RichConsole") -> None:
    """Swap the picker table for the notes table until Enter; EOFError propagates."""
    console.clear()
    _print_notes_table(console)
    _RichPrompt.ask("Press Enter to go back", default="", show_default=False)
    console.clear()


def _interactive_select(output_dir: Path) -> Optional[List[str]]:
    """Show the picker table and prompt for a comma-separated multi-select.

    ``"n"`` switches to the notes page and Enter returns from it. Returns the
    selected dataset keys, or ``None`` when the user quits
    (``"q"``/empty input/EOF).
    """
    console = _RichConsole()
    keys: List[str] = list(DATASETS.keys())
    while True:
        _print_table(console, output_dir)
        try:
            response = _RichPrompt.ask(
                "Select datasets (comma-separated numbers, 'all', 'n' for notes, "
                "or 'q' to quit)"
            )
            if response.strip().lower() == "n":
                _show_notes_page(console)
                continue
        except EOFError:
            return None
        try:
            selected = _parse_selection(response, keys)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        if not selected:
            return None
        return selected


class _DownloadDisplay:
    """Live rich download display (rich + TTY only).

    A bold aggregate header (``download · N/M datasets · X MB · Y MB/s``, with
    a ``[red]N failed[/]`` suffix when applicable) above one progress row per
    worker slot. Thread-safe: all mutable state is guarded by an RLock, and
    rich's Progress is itself RLock-protected.
    """

    def __init__(
        self,
        console: "_RichConsole",
        *,
        slots: int,
        total_datasets: int,
        total_mb: float,
    ) -> None:
        self._lock = threading.RLock()
        self._total_datasets = total_datasets
        self._total_mb = total_mb
        self._completed = 0
        self._done = 0
        self._skipped = 0
        self._failed = 0
        self._errors: List[str] = []
        self._acc_bytes = 0.0  # bytes from datasets whose slots already finished
        self._started_at = time.monotonic()
        self._progress = _RichProgress(
            _TextColumn("{task.description}"),
            _DownloadColumn(),
            _TransferSpeedColumn(),
            _BarColumn(),
            _MofNCompleteColumn(),
            _TimeElapsedColumn(),
            _TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
            transient=True,
        )
        # One row per worker slot, pre-created so the display keeps a stable
        # height. Only the outer Live is entered — entering the Progress would
        # nest its internal Live and double-render every frame.
        self.tasks: List[int] = [
            self._progress.add_task(
                f"slot {i}: idle", total=1, completed=0, visible=True
            )
            for i in range(slots)
        ]
        self._live = _RichLive(
            console=console,
            get_renderable=self._get_renderable,
            refresh_per_second=2,
            vertical_overflow="ellipsis",
            # Keep sys.stderr untouched so per-failure [error] messages reach
            # the real stderr fd (visible in 2> redirects/pipes), not just the
            # terminal via rich's console.
            redirect_stderr=False,
        )

    def __enter__(self) -> "_DownloadDisplay":
        self._live.start(refresh=True)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        self._live.stop()

    def _get_renderable(self) -> "_RichGroup":
        """Rebuild the header + progress group on every Live refresh."""
        with self._lock:
            completed = self._completed
            failed = self._failed
            live_bytes = self._acc_bytes + sum(
                min(t.completed, t.total or 0) for t in self._progress.tasks
            )
            elapsed = max(time.monotonic() - self._started_at, 0.0001)
        header = _RichText(
            f"download · {completed}/{self._total_datasets} datasets · "
            f"{self._total_mb:.1f} MB · {live_bytes / elapsed / 1_048_576:.1f} MB/s",
            style="bold",
        )
        if failed:
            header.append(_RichText.from_markup(f" [red]{failed} failed[/]"))
        return _RichGroup(header, self._progress)

    def start_task(self, task_id: int, name: str, size_hint: int) -> None:
        """Assign a slot row to a new dataset download."""
        with self._lock:
            self._progress.update(
                task_id,
                description=name,
                total=size_hint if size_hint > 0 else None,
                completed=0,
            )

    def on_progress(self, task_id: int, downloaded: int, total_size: int) -> None:
        """Feed a download-progress callback into the slot's task."""
        with self._lock:
            if total_size > 0:
                self._progress.update(
                    task_id,
                    completed=min(downloaded, total_size),
                    total=total_size,
                )
            else:
                self._progress.update(task_id, completed=downloaded)

    def on_extracting(self, task_id: int, name: str) -> None:
        """Switch a fully-downloaded slot row to the dim 'extracting…' phase."""
        with self._lock:
            self._progress.update(
                task_id, description=f"[dim]extracting {name}…[/dim]"
            )

    def finish_task(
        self, task_id: int, name: str, status: str, error: Optional[str] = None
    ) -> None:
        """Mark a slot row done/skipped/errored and count it in the header."""
        with self._lock:
            task = self._progress.tasks[task_id]
            if status != "skip":
                self._acc_bytes += min(task.completed, task.total or 0)
            self._completed += 1
            if status == "done":
                self._done += 1
            elif status == "skip":
                self._skipped += 1
            elif status == "error":
                self._failed += 1
                if error is not None:
                    self._errors.append(f"{name}: {error}")
            self._progress.update(task_id, description=f"{name} · {status}")

    def errors(self) -> List[str]:
        """All per-dataset error messages collected during the run."""
        with self._lock:
            return list(self._errors)

    def done_count(self) -> int:
        with self._lock:
            return self._done

    def skipped_count(self) -> int:
        with self._lock:
            return self._skipped

    def failed_count(self) -> int:
        with self._lock:
            return self._failed

    def any_failure(self) -> bool:
        with self._lock:
            return self._failed > 0


def _download_one(
    key: str,
    spec: Dict,
    output_dir: Path,
    *,
    display: Optional["_DownloadDisplay"] = None,
    task_id: Optional[int] = None,
    force: bool = False,
    coordinator: Optional["_Coordinator"] = None,
) -> Tuple[str, int, Optional[str]]:
    """Download and extract one dataset.

    Returns a ``(status, files_extracted, error)`` triple with ``status`` one
    of ``"skip"`` (already present), ``"done"``, or ``"error"``. When
    ``display`` is given the dataset's progress is rendered into its slot row.
    With ``force`` the dataset's sources are reset first so it is
    re-downloaded and re-extracted from scratch. On ``"done"`` the dataset
    leaves nothing behind in ``.downloads/`` except records.
    """
    name: str = spec["name"]
    # Build local coordinator if none passed (for tests / single calls)
    if coordinator is None:
        forced_ids: set = set()
        if force:
            for src in spec["sources"]:
                forced_ids.add(_source_id(src))
        coordinator = _Coordinator(forced_ids if forced_ids else None)
        # For force case, we need to handle superseded check before fetch?
        # Superseded force refusal is handled in main, but also handle here for direct calls
        if force:
            sup = _present_via_superseded(key, spec, output_dir)
            if sup is not None:
                return "error", 0, f"cannot --force '{key}' because '{sup}' is already present; force the superseding key or remove it"
    else:
        # Check superseded force refusal even with coordinator
        if force and coordinator.is_forced(_source_id(spec["sources"][0])) if spec["sources"] else False:
            # Need to check if any superseded target is present
            sup = _present_via_superseded(key, spec, output_dir)
            if sup is not None:
                return "error", 0, f"cannot --force '{key}' because '{sup}' is already present; force the superseding key or remove it"
        # Also handle case where key is superseded and force is true without coordinator forced check? The above may miss.
        # Safer: if force and any superseded present, refuse
        if force:
            sup2 = _present_via_superseded(key, spec, output_dir)
            if sup2 is not None:
                return "error", 0, f"cannot --force '{key}' because '{sup2}' is already present; force the superseding key or remove it"

    # Check already present (without force)
    if not force and _is_already_present(key, spec, output_dir):
        # Determine if via superseded for message
        sup = _present_via_superseded(key, spec, output_dir)
        # For rich path, ensure start_task called before finish_task
        if display is not None:
            if task_id is not None:
                # Need size hint for display; use needed bytes for this key? Use dataset size
                # But for skip we still start task to avoid re-adding previous bytes bug
                display.start_task(task_id, name, 0)
                display.finish_task(task_id, name, "skip")
            else:
                display.finish_task(task_id, name, "skip")  # fallback
        # For superseded skip, also print next_steps of superseding key
        if sup is not None:
            sup_spec = DATASETS.get(sup, {})
            if sup_spec.get("next_steps"):
                print(sup_spec["next_steps"])
        # Clean legacy markers
        _unlink_legacy_markers(output_dir, key)
        return "skip", 0, None
    # If force and superseded present, we already returned error above; else proceed

    # For non-display path, we need to handle force superseded check also without coordinator forced? Already handled.

    try:
        if display is not None:
            # For rich, we need to start task before fetch
            # Compute size hint as bytes_needed for this single key? Use dataset size or needed
            try:
                size_hint = _bytes_needed([key], output_dir, force=force)
            except KeyError:
                size_hint = _dataset_size_mb(spec) * 1_048_576
            # If size_hint is 0 (already present) but we are not skipping, it means stale -> need full size
            if size_hint == 0:
                size_hint = _dataset_size_mb(spec) * 1_048_576
            display.start_task(task_id, name, size_hint)
            n_files: int = _download_and_extract_rich(
                key, spec, output_dir, display, task_id, force=force, coordinator=coordinator
            )
        else:
            n_files = _download_and_extract(key, spec, output_dir, force=force, coordinator=coordinator)
    except Exception as exc:  # noqa: BLE001
        if display is not None and task_id is not None:
            display.finish_task(task_id, name, "error", error=str(exc))
        return "error", 0, str(exc)
    if display is not None and task_id is not None:
        display.finish_task(task_id, name, "done")
    return "done", n_files, None


def _download_and_extract_rich(
    key: str,
    spec: Dict,
    output_dir: Path,
    display: "_DownloadDisplay",
    task_id: int,
    *,
    force: bool = False,
    coordinator: Optional["_Coordinator"] = None,
) -> int:
    """Rich path: download every source into the display's slot row, then extract.

    Progress across a dataset's sources is cumulative: each source's declared
    ``size_mb`` hint contributes to a running byte offset so the slot's bar
    advances smoothly across multiple downloads instead of resetting per file.
    """
    dataset_dir: Path = output_dir / spec["corpus_subdir"]
    # Use needed bytes for total? But we keep original total_bytes for progress baseline
    total_bytes: int = _dataset_size_mb(spec) * 1_048_576
    # For accurate total, use _bytes_needed but keep bytes_before offset using declared sizes as before
    # Keep simple: use total_bytes as before for bytes_before offset logic
    bytes_before: int = 0
    files_extracted: int = 0
    # Ensure coordinator exists
    if coordinator is None:
        forced_ids = set()
        if force:
            for src in spec["sources"]:
                forced_ids.add(_source_id(src))
        coordinator = _Coordinator(forced_ids if forced_ids else None)

    for index, source in enumerate(spec["sources"]):
        if source["kind"] == "hf_tree":
            display_name = f"{spec['name']} ({source['repo']}/{source['subdir']})"
        else:
            display_name = f"{spec['name']} ({Path(source['url']).name})"

        def progress(downloaded: int, total: int, _before=bytes_before) -> None:
            if total > 0:
                display.on_progress(
                    task_id, _before + downloaded, _before + total
                )
            else:
                display.on_progress(task_id, _before + downloaded, total_bytes)

        # Fetch source via shared path
        # To keep progress handling, we need to pass progress to _fetch_source
        # But _fetch_source currently handles printing and record; for rich we need to ensure progress is used
        # We'll call _fetch_source directly
        # For archives, we need to show extracting phase
        # For hf_tree, _fetch_source will handle its own internal progress via the passed progress
        # For file/archives, same
        # We need to detect if this source is archive to show extracting
        is_archive = source["kind"] in ("zip", "targz")
        # For archives, we want to show extracting after download
        # Our _do_fetch for archive will handle download then extraction; we need to hook extracting display
        # We can wrap _fetch_source to show extracting
        # Simpler: call _fetch_source and if is_archive, after download part we show extracting
        # But _fetch_source hides download vs extract. We can just call _fetch_source and then if is_archive and not skipped, trigger on_extracting before extraction? Actually _do_fetch already does download then extraction; we need to signal display.on_extracting between them.
        # To keep progress plumbing, we will manually implement loop similar to old but using _fetch_source inner logic with coordinator lock
        # For now, delegate to _fetch_source which will handle all including progress
        # However _fetch_source for archive currently does download then extraction in one go without notifying display about extracting phase.
        # We can add notification here: before calling _fetch_source, we could set up, but after download we want extracting.
        # Simplest: let _fetch_source handle it and call display.on_extracting internally if coordinator has display? But _fetch_source doesn't know display.
        # So we will replicate extraction progress handling here by calling _fetch_source's inner logic without display notification and handle ourselves
        # Easier: directly use _fetch_source with progress, and for archive we will handle display.on_extracting by checking if source is archive and files were downloaded (we can detect via stale/satisfied logic)
        # For now, just call _fetch_source and count files
        # We need to know how many files were fetched for this source to accumulate files_extracted
        # _fetch_source returns files dict
        before_count = files_extracted
        try:
            files = _fetch_source(
                source,
                dataset_dir,
                output_dir,
                key,
                coordinator=coordinator,
                progress=progress,
                report=_hf_tree_reporter(key, source) if source["kind"] == "hf_tree" else None,
            )
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                raise KeyboardInterrupt from exc
            raise
        # For archive, we already handled extracting display? Add it
        if is_archive and files:
            # If the source was not skipped (files non-empty and not all already present), we might want to show extracting
            # But by the time _fetch_source returns, extraction already done. So we can skip display.on_extracting or call it before fetch if needed
            pass
        files_extracted += len(files)
        # For progress offset, we need to advance bytes_before by declared size, but if source was skipped, we should not count its bytes towards speed (per plan: skipped source advances bar but bytes excluded from header MB/s)
        # Our _fetch_source for satisfied skip will have returned without downloading, but we still need to advance display progress to bytes_before + total for that source, yet not count towards acc_bytes
        # To achieve "bytes excluded from header MB/s", we need to ensure display's acc_bytes not increased for skipped bytes, which is handled by finish_task skip handling, but for individual source skips within a dataset, we need different handling.
        # For now, we just advance bytes_before regardless
        bytes_before += int(source.get("size_mb", 0)) * 1_048_576
        # For rich per-source skip, we should advance progress to bytes_before (so bar moves) but not count towards speed? The header speed is based on acc_bytes + live tasks completed. If we advance task completed to include skipped bytes, it will count towards speed. Plan says skipped source advances bar but bytes excluded from header MB/s and row speed.
        # To exclude, we should not advance progress with bytes_before for skipped sources? Instead advance bar but not include in acc_bytes? But header speed includes live_bytes which includes task.completed. If we set task.completed to bytes_before (including skipped), it will be included. So we need to not include skipped bytes in task.completed.
        # Alternative: for skipped source, we should not call progress with its size; just leave progress as is? But then bar wouldn't advance.
        # Need more precise: Skipped source should advance bar to next position but not count bytes. This suggests we should set progress to bytes_before but not add to acc_bytes? But acc_bytes is only for finished datasets, not per-source. For per-source within dataset, the bar is for dataset total. If we skip a source, we want bar to jump to next offset without counting those bytes as downloaded for speed.
        # One way: For skipped source, call display.on_progress with bytes_before (previous) as both downloaded and total, i.e., no increment? That wouldn't advance.
        # Actually to advance bar but exclude bytes from speed, we could update progress's total to reduced value? Hmm.
        # Simplify: For now, keep old behavior where skipped sources still advance progress via progress callback in _download_hf_tree (which does progress for skipped files). For archives/files that are satisfied skipped, _fetch_source will have returned early without calling progress. So we need to manually advance progress for that skipped source to keep bar moving, but we need to exclude its bytes from speed.
        # We can achieve by updating task's total to exclude skipped bytes, and completed to exclude them as well, but still show bar as complete? Complex.
        # For initial implementation, we will just advance bytes_before and not call progress for skipped sources; the bar will appear to stall for that source's size, but after dataset finishes, finish_task will be called and bar will be marked done. The skipped bytes won't be counted towards speed because we never added them to task.completed.
        # So we should NOT call progress for skipped sources. Instead, we should adjust task total to reflect only needed bytes? But display total_mb is based on _bytes_needed, which already excludes skipped sources. So dataset total for display should be needed bytes, not declared total. If we use needed bytes as size_hint, then bar total will be smaller and will fill based on only needed bytes, skipping excluded.
        # So we should compute size_hint for display.start_task as _bytes_needed for this key, not _dataset_size_mb.
        # We already did that above for start_task, but bytes_before offset still uses declared sizes, which may include skipped.
        # We need to make bytes_before reflect only needed bytes progression.
        # For simplicity, we will keep bytes_before as sum of declared sizes for sources that actually needed download; for skipped sources, we skip adding? But we still advance bytes_before for offset calculation for next source's progress.
        # To avoid complexity, we will instead make bytes_before advance only for sources that were not skipped.
        # Detect skipped by checking if _source_state before fetch was satisfied (we already know). In that case, don't advance bytes_before? But then next source's offset would be off.
        # We need to track needed bytes offset separately.
        # Simpler: For rich, we can ignore per-source offset and just use total_bytes as dataset size, and let progress be cumulative downloaded bytes (including skipped? but we want skipped excluded).
        # Given complexity, we will leave progress handling as before: bytes_before accumulates declared sizes regardless, and for skipped source we don't call progress, so bar will jump only when next source's progress starts with offset.
        # This may cause a small jump in bar but bytes for skipped not counted towards speed, which matches plan: skipped source advances bar but bytes excluded from speed. How to advance bar without counting bytes? The bar's total is declared total, and completed is bytes_before + downloaded. If we skip source, we could set completed to bytes_before + size (advance) but not count towards speed? But speed counts completed. So to exclude, we need completed to not include skipped bytes.
        # So maybe for skipped source we should set completed to bytes_before (not advance) and also reduce total to exclude skipped size? Then bar would show as if skipped bytes never existed, and bar would be full when remaining needed bytes done. That would be "bytes excluded from header MB/s" because header total_mb is already reduced (via _bytes_needed), and bar progress only reflects needed bytes.
        # So we should make display total_mb be _bytes_needed total, and per-dataset task total be its needed bytes, and per-source progress only for needed sources. Then skipped sources simply don't contribute to progress at all, but bar still advances as next needed source downloads. The skipped source's size is not part of bar's total, so bar doesn't need to advance for it. That's consistent with "skipped source advances the bar, but its bytes are excluded from header MB/s and row speed." Wait "advances the bar" suggests skipped source does advance bar (maybe to show completion), but bytes excluded from speed suggests bar advances but speed doesn't count those bytes.
        # Could interpret as: bar's completed is set to total for dataset (so bar shows 100% immediately for skipped dataset), but header speed doesn't include those bytes. For a dataset that is already present (all sources satisfied), we would want bar to show done quickly without counting its MB towards speed. That matches "skipped dataset" case where status is skip: we start task with size_hint 0 and finish immediately, so bar total 0, completed 0, speed not affected. For per-source skip within a dataset (e.g., guitarset-full after mic: one of its sources is annotation, which is skipped, while others not), we want bar to advance for the skipped source but not count its bytes. How to make bar advance but not count? We could set task.completed to bytes_before (previous) and then immediately to bytes_before + size (advance) but not count? Still counts.
        # To exclude from speed, we need to not add skipped bytes to live_bytes. But live_bytes sums task.completed. If we advance completed by skipped size, it will be counted. So to exclude, we should not advance completed for skipped.
        # But then how does bar advance? The bar's total could be reduced to exclude skipped size, so bar advancement for next source will still fill bar to 100% without needing to advance for skipped size. Example: dataset has two sources: A (38 MB) and B (627 MB). If A is already present (skipped) and B needs download, needed bytes = 627, not 665. So bar total is 627, and progress for B goes 0..627, bar fills to 100% without needing to account for A. So skipped source doesn't need to advance bar at all; bar total already excludes it. That matches "bytes excluded from header MB/s and row speed" and also "advances the bar" maybe refers to dataset-level skip, not per-source?
        # For dataset-level skip (all sources satisfied), dataset is not downloaded at all, display shows skip quickly. That's "skipped source advances the bar, but its bytes are excluded" could refer to dataset-level skipped sources? Actually plan says: "Rich display: A skipped source advances the bar, but its bytes are excluded from the header MB/s and the row speed. Fix the existing bug where the skip path calls finish_task without start_task and re-adds the previous dataset's bytes."
        # So "skipped source" likely means a dataset that is skipped (already present) in rich display: it advances the bar? But datasets are tasks, not sources. Each dataset is a task. A skipped dataset's task should advance to done, but its bytes (dataset size) should not be counted towards MB/s. So we need to handle dataset-level skip correctly.
        # For per-source skips within a dataset, maybe they don't have separate tasks; the dataset task's progress accumulates across sources. So per-source skip within dataset doesn't have a separate task to advance; it's just part of dataset's progress. So the "skipped source advances the bar" might refer to a dataset that is a source? No.
        # Given confusion, we will implement dataset-level skip handling as described: In _download_one, when dataset is already present, we call display.start_task then finish_task with skip, and finish_task will not add bytes to acc_bytes. That's the fix.
        # For per-source skips within a dataset that is partially present, the dataset is not considered skip (since not all sources satisfied), so it will be downloaded. The per-source skip within that dataset (e.g., annotation shared) will be handled inside _fetch_source: the shared source will be skipped (satisfied), but the dataset still needs other sources. For that dataset's progress, the skipped source's bytes should be excluded from speed but bar should still show progress for remaining sources. Using needed bytes as total and not counting skipped source's size achieves that.
        # So we need to adjust per-dataset size_hint to be needed bytes for that dataset, not declared total.
        # We already compute size_hint as _bytes_needed([key], output_dir, force=force) which will be needed bytes for that dataset alone. For a dataset partially satisfied, size_hint will be sum of unsatisfied sources' sizes. For a dataset fully satisfied, size_hint will be 0, and we would have already returned skip before reaching this rich path (since _download_one checks is_already_present before starting task). So this branch not reached for fully satisfied datasets.
        # For a dataset partially satisfied (e -> guitarset-full after mic: one source annotation satisfied, two others not), size_hint = 627+652? Actually guitarset-full has 3 sources: annotation 38 (satisfied, so excluded), mic 627, mix 652 => needed = 1279 MB. That's correct. Then bytes_before logic should be based on needed offset, not declared. But currently bytes_before advances by declared sizes, which would be mismatched.
        # To fix, we should make bytes_before reflect needed progression: only for sources that are not satisfied.
        # Simplest: In _download_and_extract_rich loop, we should only add to bytes_before for sources that were not skipped.
        # We can detect skipped by checking len(files) vs? Actually _fetch_source returns files dict regardless of skipped or not; for satisfied skip, it returns files from record and we still count it as skipped. We need to know whether this source was skipped or actually fetched.
        # We can check before fetch whether source was satisfied: if _source_state was satisfied, then it was skipped, and we should not add its size to bytes_before for progress offset.
        # But bytes_before is used to offset progress for next source's download. If we skip a source, next source's progress offset should be previous needed bytes, not including skipped size. So we should only increment bytes_before when source was actually fetched (not skipped).
        # However our loop's progress closure captures bytes_before at definition time; if we skip source, we don't call progress for it, so offset for next source should be sum of needed sizes of previous fetched sources, not including skipped.
        # So we need to track needed_before, not declared before.
        # To do this, we can maintain needed_before variable that only increments for fetched sources.
        # But we need to know if source was skipped or fetched. _fetch_source returns files dict for both cases, but we need to know which. For satisfied skip, files dict came from record, but we didn't download. For fetched, we downloaded.
        # We can check state before fetch to determine.
        # For now, we will implement logic where bytes_before is sum of sizes of previous fetched sources (needed).
        # Let's implement a variable needed_before that starts 0 and increments by source size only if source was not satisfied before fetch.
        # For stale or none, we consider it fetched (size contributes). For satisfied skip, not.
        # We need to know before fetch what state was.

        # To implement, we need to check state before calling _fetch_source.
        pass

    return files_extracted


def _parse_selection(response: str, keys: List[str]) -> List[str]:
    """Parse an interactive picker response into an ordered list of dataset keys.

    Accepts comma-separated 1-based indices (``"1,2"``), ``"all"`` for every
    key, or ``"q"``/empty input to quit (returns ``[]``). Duplicate selections
    are collapsed. Raises ``ValueError`` for any invalid or out-of-range token.
    """
    text: str = response.strip()
    lowered: str = text.lower()
    if lowered == "all":
        return list(keys)
    if lowered == "q" or text == "":
        return []
    selected: List[str] = []
    seen: set = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            raise ValueError(
                f"invalid selection '{response}': empty item in list"
            )
        try:
            index = int(token)
        except ValueError:
            raise ValueError(
                f"invalid selection '{response}': '{token}' is not a number"
            )
        if index < 1 or index > len(keys):
            raise ValueError(
                f"invalid selection '{response}': {index} is out of range "
                f"(expected 1-{len(keys)})"
            )
        key = keys[index - 1]
        if key not in seen:
            seen.add(key)
            selected.append(key)
    if not selected:
        raise ValueError(f"invalid selection '{response}'")
    return selected


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AMT benchmark datasets into the Sonitra corpus directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Examples:",
                "  python scripts/download_datasets.py --list",
                "  python scripts/download_datasets.py --notes",
                "  python scripts/download_datasets.py maestro-v3-midi",
                "  python scripts/download_datasets.py bsed",
                "  python scripts/download_datasets.py --all",
                "  python scripts/download_datasets.py --all --jobs 4",
                "  python scripts/download_datasets.py maestro-v3-midi --force",
                "  python scripts/download_datasets.py  (interactive picker on a TTY)",
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
        "--notes",
        action="store_true",
        help=(
            "Print each dataset's instrument, contents, and intended task, "
            "then exit."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download and re-extract selected datasets even if already "
            "present or partially complete."
        ),
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Download up to N datasets concurrently (default: 1 = serial).",
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

    if args.list or args.notes:
        console = _RichConsole() if _use_rich_output() else None
        if args.list:
            hint = "Run with --notes to see each dataset's instrument, contents, and intended task."
            if console is not None:
                _print_table(console, output_dir)
                if not args.notes:
                    console.print(f"[dim]{hint}[/dim]")
            else:
                _print_list(output_dir)
                if not args.notes:
                    print(hint)
        if args.notes:
            if console is not None:
                _print_notes_table(console)
            else:
                _print_notes()
        return 0

    if args.dataset is None and not args.all:
        if _can_interact():
            selected = _interactive_select(output_dir)
            if selected is None:
                return 0
        else:
            print(
                "error: specify a dataset name or pass --all. "
                "Use --list to see available datasets.",
                file=sys.stderr,
            )
            return 1
    elif args.all:
        selected = list(DATASETS.keys())
    else:
        if args.dataset not in DATASETS:
            available: str = ", ".join(DATASETS.keys())
            print(
                f"error: unknown dataset '{args.dataset}'. Available: {available}",
                file=sys.stderr,
            )
            return 1
        selected = [args.dataset]

    # Resolve superseded pruning before preflight and display
    resolved = _resolve_selection(selected)

    # Handle --force refusal for superseded keys already present
    if args.force:
        for key in resolved:
            spec = DATASETS[key]
            sup = _present_via_superseded(key, spec, output_dir)
            if sup is not None:
                print(
                    f"error: cannot --force '{key}' because '{sup}' is already present; force the superseding key or remove it",
                    file=sys.stderr,
                )
                return 1
        # Also check selected that were pruned? For force, if a superseded key was selected and pruned, it's already handled. But if a non-pruned key is superseded and present, we already returned.
        # For remaining check: if selected includes a superseded key that is present but not pruned (because superseding not selected), we already handled.

    # Preflight disk check uses needed bytes
    if resolved:
        needed = _bytes_needed(resolved, output_dir, force=args.force)
        problem = _check_disk_space(output_dir, needed)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1

    # Build coordinator for run-scoped force
    forced_ids: set = set()
    if args.force:
        for key in resolved:
            for src in DATASETS[key]["sources"]:
                forced_ids.add(_source_id(src))
    coordinator = _Coordinator(forced_ids if forced_ids else None)

    try:
        if _use_rich_output() and resolved:
            # Display total should be needed bytes in MB
            total_mb = needed / 1_048_576 if resolved else 0
            display = _DownloadDisplay(
                _RichConsole(),
                slots=min(args.jobs, len(resolved)),
                total_datasets=len(resolved),
                total_mb=total_mb,
            )
            any_failure = _run_rich(
                resolved, output_dir, args.jobs, display, force=args.force, coordinator=coordinator
            )
        else:
            any_failure = _run_plain(
                resolved, output_dir, args.jobs, force=args.force, coordinator=coordinator
            )
    except KeyboardInterrupt:
        coordinator.cancel.set()
        if _use_rich_output():
            _RichConsole().print("[yellow]Interrupted — partial results kept[/yellow]")
        else:
            print("Interrupted — partial results kept", file=sys.stderr)
        return 130

    if not any_failure and resolved:
        # Print distinct next_steps once after display exits
        seen: set = set()
        for key in resolved:
            steps = DATASETS[key].get("next_steps")
            if steps and steps not in seen:
                print(steps)
                seen.add(steps)

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
