"""Discover corpus files and pair each recording with its reference MIDI file."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

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
        ambiguous: The subset of *unpaired_audio* that stopped on two or more
            candidates, mapped to those candidates (sorted), so callers can
            tell "no match" from "ambiguous" without re-running the match.
    """

    mapping: dict[Path, Path] = field(default_factory=dict)
    unpaired_audio: list[Path] = field(default_factory=list)
    unpaired_midi: list[Path] = field(default_factory=list)
    ambiguous: dict[Path, list[Path]] = field(default_factory=dict)


def _tokens(path: Path) -> list[str]:
    return path.stem.split("_")


def match_token_prefix(
    query_tokens: list[str],
    candidates: dict[T, list[str]],
) -> tuple[T | None, list[T], int]:
    """Pure token-prefix matcher.

    Deterministic, top-down token-prefix matching. Tokens are ``_``-split
    stems, compared case-sensitively. ``k`` descends from
    ``min(len(query_tokens), max(len(candidate_tokens)))`` down to ``1``.
    At each ``k``, ``candidates_k = {c : len(c_tokens) >= k and
    c_tokens[:k] == query_tokens[:k]}``:

    - exactly one candidate -> unique match, stop.
    - two or more candidates -> ambiguous, stop (smaller ``k`` can only
      grow the set further, never disambiguate).
    - zero candidates -> continue to ``k - 1``.

    This is the pure core of :func:`pair_audio_to_reference`, extracted so
    that the same logic can be reused for metadata joins without
    duplicating the descending-``k`` loop.

    Args:
        query_tokens: ``_``-split stem of the query (audio stem or song).
        candidates: Mapping from candidate key to its ``_``-split tokens.
            For audio pairing the key is a :class:`Path` to a MIDI file;
            for metadata joins the key is the metadata stem string.

    Returns:
        A tuple ``(match, candidates_at_stop, k)`` where ``match`` is the
        unique candidate key or ``None``, ``candidates_at_stop`` is the
        list of candidates at the ``k`` where the search stopped (empty
        when no candidate ever matched), and ``k`` is that prefix length
        (``0`` when no candidate ever matched).  ``candidates_at_stop``
        and ``k`` are returned so callers can distinguish "no match" from
        "ambiguous" and emit the same warning as
        :func:`pair_audio_to_reference`.

    Example:
        >>> from pathlib import Path
        >>> cands = {Path("1727"): ["1727"], Path("1728"): ["1728"]}
        >>> match_token_prefix(["1727", "schubert", "op114", "2"], cands)
        (PosixPath('1727'), [PosixPath('1727')], 1)
    """
    max_r_len = max((len(tokens) for tokens in candidates.values()), default=0)
    k_start = min(len(query_tokens), max_r_len)
    for k in range(k_start, 0, -1):
        candidates_k: list[T] = [
            key
            for key, tokens in candidates.items()
            if len(tokens) >= k and tokens[:k] == query_tokens[:k]
        ]
        # Deterministic ordering regardless of dict insertion order.
        candidates_k = sorted(candidates_k, key=lambda x: str(x))
        if len(candidates_k) == 1:
            return candidates_k[0], candidates_k, k
        if len(candidates_k) >= 2:
            return None, candidates_k, k
        # zero candidates -> continue descending
    return None, [], 0


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

    mapping: dict[Path, Path] = {}
    unpaired_audio: list[Path] = []
    ambiguous_candidates: dict[Path, list[Path]] = {}

    for audio in sorted(audio_paths):
        a_tokens = _tokens(audio)
        matched, candidates_at_stop, k = match_token_prefix(a_tokens, midi_tokens)

        if matched is not None:
            mapping[audio] = matched
        else:
            if candidates_at_stop:
                # Ambiguous: stopped on two or more candidates.
                # Sort for deterministic reporting (match_token_prefix already sorts).
                ambiguous_candidates[audio] = candidates_at_stop
                logger.warning(
                    "Ambiguous audio-to-reference pairing for %s at k=%d: %s",
                    audio,
                    k,
                    [str(c) for c in candidates_at_stop],
                )
            else:
                logger.warning("No reference MIDI found for audio file %s", audio)
            unpaired_audio.append(audio)

    paired_midi = set(mapping.values())
    unpaired_midi = [midi for midi in midi_paths if midi not in paired_midi]

    return PairingResult(
        mapping=mapping,
        unpaired_audio=unpaired_audio,
        unpaired_midi=unpaired_midi,
        ambiguous=ambiguous_candidates,
    )
