"""Corpus discovery and audio-to-reference-MIDI pairing.

Sonitra's audio-input mode reads source recordings from a dataset's
``recordings/`` directory and pairs each one to a reference MIDI in the
sibling ``midi/`` directory (see ``scripts/download_datasets.py`` for the
BSED layout this mirrors). This module is the single, reusable home for that
discovery + pairing logic; ``sonitra.cli._discover_midi_files`` re-exports
``discover_midi_files`` rather than duplicating the walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_MIDI_SUFFIXES: frozenset[str] = frozenset({".mid", ".midi"})
_AUDIO_SUFFIXES: frozenset[str] = frozenset({".wav", ".flac", ".mp3"})


def discover_midi_files(directory: Path) -> list[Path]:
    """Recursively find ``.mid``/``.midi`` files under *directory*.

    Args:
        directory: Root directory to walk.

    Returns:
        Sorted list of matching file paths (case-insensitive extension
        match; directories are ignored, even ones named like a MIDI file).
    """
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _MIDI_SUFFIXES
    )


def discover_audio_files(directory: Path) -> list[Path]:
    """Recursively find ``.wav``/``.flac``/``.mp3`` files under *directory*.

    Args:
        directory: Root directory to walk.

    Returns:
        Sorted list of matching file paths (case-insensitive extension
        match; directories are ignored, even ones named like an audio file).
    """
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_SUFFIXES
    )


@dataclass(frozen=True)
class PairingResult:
    """Outcome of :func:`pair_audio_to_reference`.

    Attributes:
        mapping: Audio path -> paired reference MIDI path, for every audio
            file that found exactly one unique match.
        unpaired_audio: Audio files that never found a unique reference
            (either no candidate matched at any prefix length, or the match
            was ambiguous).
        unpaired_midi: Reference MIDI files that no audio file paired to.
    """

    mapping: dict[Path, Path] = field(default_factory=dict)
    unpaired_audio: list[Path] = field(default_factory=list)
    unpaired_midi: list[Path] = field(default_factory=list)


def _tokens(path: Path) -> list[str]:
    return path.stem.split("_")


def pair_audio_to_reference(
    audio_paths: Sequence[Path],
    midi_paths: Sequence[Path],
) -> PairingResult:
    """Pair each audio file to its unique reference MIDI by token prefix.

    Deterministic, top-down token-prefix matching. Tokens are the ``_``-split,
    extension-stripped stem, compared case-sensitively. For an audio file
    with tokens ``A`` and candidate references each with tokens ``R``, ``k``
    descends from ``min(len(A), max(len(R) for all candidates))`` down to
    ``1``. At each ``k``, ``candidates_k = {R : len(R) >= k and R[:k] ==
    A[:k]}``:

    - exactly one candidate -> pair, stop (smaller ``k`` is not consulted).
    - two or more candidates -> ambiguous, unpaired + warning, stop (smaller
      ``k`` can only grow the candidate set further, never disambiguate).
    - zero candidates -> continue to ``k - 1``.

    If ``k`` reaches ``0`` without ever producing a unique match, the audio
    file is unpaired.

    Args:
        audio_paths: Source audio files to pair.
        midi_paths: Candidate reference MIDI files.

    Returns:
        A :class:`PairingResult` with the mapping plus both unpaired lists.
    """
    midi_paths = sorted(midi_paths)
    midi_tokens = {midi: _tokens(midi) for midi in midi_paths}
    max_r_len = max((len(tokens) for tokens in midi_tokens.values()), default=0)

    mapping: dict[Path, Path] = {}
    unpaired_audio: list[Path] = []

    for audio in sorted(audio_paths):
        a_tokens = _tokens(audio)
        k_start = min(len(a_tokens), max_r_len)

        matched: Path | None = None
        ambiguous = False
        for k in range(k_start, 0, -1):
            candidates = [
                midi
                for midi, r_tokens in midi_tokens.items()
                if len(r_tokens) >= k and r_tokens[:k] == a_tokens[:k]
            ]
            if len(candidates) == 1:
                matched = candidates[0]
                break
            if len(candidates) >= 2:
                ambiguous = True
                logger.warning(
                    "Ambiguous audio-to-reference pairing for %s at k=%d: %s",
                    audio,
                    k,
                    [str(c) for c in candidates],
                )
                break
            # zero candidates at this k -> keep descending.

        if matched is not None:
            mapping[audio] = matched
        else:
            if not ambiguous:
                logger.warning("No reference MIDI found for audio file %s", audio)
            unpaired_audio.append(audio)

    paired_midi = set(mapping.values())
    unpaired_midi = [midi for midi in midi_paths if midi not in paired_midi]

    return PairingResult(
        mapping=mapping,
        unpaired_audio=unpaired_audio,
        unpaired_midi=unpaired_midi,
    )
