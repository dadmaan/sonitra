"""Resolve the config ``io`` selection keys into the corpus files a command processes.

Filters reference MIDI by dataset metadata, pairs audio through the corpus
token-prefix matcher, and samples deterministically. Returns data only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import logging
from pathlib import Path
import random
from typing import Any, Literal, Mapping, Sequence

from sonitra.config import ConfigError, PipelineConfig, SelectionSample
from sonitra.corpus import pair_audio_to_reference

logger = logging.getLogger(__name__)


class SelectionError(ConfigError):
    """A selection cannot be resolved against the corpus."""


@dataclass(frozen=True)
class SelectionResult:
    """Resolved units plus flat provenance inputs for the summary block."""

    units: list[Path]
    references: list[Path]
    unit_kind: Literal["reference_midi", "recording", "audio"]
    counts: dict[str, int]
    by_value: dict[str, dict[str, int]]
    unmatched: list[Path]
    unpaired_audio: list[Path]
    notice: str | None
    configured: bool
    dataset: str | None
    metadata_csv: str | None
    metadata_sha256: str | None
    join_column: str
    where: dict[str, list[str]]
    sample: dict[str, int] | None

    def provenance(self, *, unit_root: Path) -> dict[str, Any]:
        files_sha256 = _files_sha256(self.units, unit_root)
        if not self.configured:
            return {
                "configured": False,
                "unit": self.unit_kind,
                "counts": {
                    "discovered": self.counts["discovered"],
                    "selected": self.counts["selected"],
                },
                "files_sha256": files_sha256,
            }
        return {
            "configured": True,
            "dataset": self.dataset,
            "metadata_csv": self.metadata_csv,
            "metadata_sha256": self.metadata_sha256,
            "join_column": self.join_column,
            "where": self.where,
            "sample": self.sample,
            "unit": self.unit_kind,
            "counts": dict(self.counts),
            "by_value": self.by_value,
            "files_sha256": files_sha256,
        }

    def summary_line(self) -> str:
        parts: list[str] = []
        if self.dataset is not None:
            parts.append(self.dataset)
        for column in sorted(self.where):
            parts.append(f"{column}={'+'.join(self.where[column])}")
        if self.sample is not None:
            parts.append(f"sample n={self.sample['n']} seed={self.sample['seed']}")
        label = " ".join(parts) or "none"
        return (
            f"selection: {label} -> "
            f"{self.counts['selected']}/{self.counts['discovered']} files"
        )

    def with_sampled_units(self, units: Sequence[Path]) -> "SelectionResult":
        sampled = list(units)
        return replace(
            self,
            units=sampled,
            references=(
                list(sampled) if self.unit_kind == "reference_midi" else self.references
            ),
            counts={**self.counts, "selected": len(sampled)},
        )


@dataclass(frozen=True)
class _ReferenceSelection:
    refs_all: list[Path]
    refs_selected: list[Path]
    unmatched: list[Path]
    metadata: dict[str, dict[str, str]] | None
    metadata_path: Path | None


def resolve_metadata_csv(cfg: PipelineConfig) -> Path:
    if cfg.io.metadata_csv is None:
        raise SelectionError(
            "io.metadata_csv is required to resolve dataset metadata"
        )
    raw = Path(cfg.io.metadata_csv)
    base = Path(cfg.io.corpus_root) / str(cfg.io.dataset) / "metadata"
    resolved = raw if raw.is_absolute() else base / raw
    if not resolved.exists():
        message = f"selection metadata CSV not found: {resolved}"
        if not raw.is_absolute() and raw.parent != Path("."):
            message = (
                f"{message} (io.metadata_csv is a bare filename resolved "
                f"under {base}); do not pass a repo-relative path"
            )
        raise SelectionError(message)
    return resolved


def _missing_column_message(column: str, path: Path, fieldnames: list[str]) -> str:
    return f"column '{column}' not found in {path} (columns: {', '.join(fieldnames)})"


def load_selection_metadata(
    path: Path,
    join_column: str,
    where: Mapping[str, list[str]],
) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if join_column not in fieldnames:
            raise SelectionError(_missing_column_message(join_column, path, fieldnames))
        for column in where:
            if column not in fieldnames:
                raise SelectionError(_missing_column_message(column, path, fieldnames))
        rows = list(reader)

    for column, values in where.items():
        available = sorted({row[column] for row in rows if row.get(column)})
        known = set(available)
        unknown = sorted({value for value in values if value not in known})
        if unknown:
            rendered = ", ".join(repr(value) for value in unknown)
            raise SelectionError(
                f"selection value(s) {rendered} not found in column '{column}' "
                f"(available: {', '.join(available)})"
            )

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = Path(row[join_column]).stem
        groups.setdefault(key, []).append(row)

    index: dict[str, dict[str, str]] = {}
    for key, group in groups.items():
        if len(group) > 1:
            for column in sorted(where):
                distinct = {row.get(column) for row in group}
                if len(distinct) > 1:
                    raise SelectionError(
                        f"metadata key '{key}' appears {len(group)} times "
                        f"with different '{column}' values"
                    )
        index[key] = group[0]
    return index


def sample_units(files: Sequence[Path], sample: SelectionSample | None) -> list[Path]:
    if sample is None:
        return list(files)
    if sample.n >= len(files):
        return sorted(files)
    return sorted(random.Random(sample.seed).sample(sorted(files), sample.n))


def _filter_references(
    cfg: PipelineConfig,
    references: Sequence[Path],
) -> _ReferenceSelection:
    refs_all = list(references)
    if not cfg.io.where:
        return _ReferenceSelection(refs_all, list(refs_all), [], None, None)

    metadata_path = resolve_metadata_csv(cfg)
    metadata = load_selection_metadata(
        metadata_path,
        cfg.io.join_column,
        cfg.io.where,
    )
    unmatched: list[Path] = []
    refs_selected: list[Path] = []
    for reference in refs_all:
        row = metadata.get(reference.stem)
        if row is None:
            unmatched.append(reference)
            continue
        if all(row.get(column) in values for column, values in cfg.io.where.items()):
            refs_selected.append(reference)
    return _ReferenceSelection(refs_all, refs_selected, unmatched, metadata, metadata_path)


def _selection_provenance(
    cfg: PipelineConfig,
    metadata_path: Path | None,
) -> dict[str, Any]:
    if not cfg.io.where and cfg.io.sample is None:
        return {
            "configured": False,
            "dataset": None,
            "metadata_csv": None,
            "metadata_sha256": None,
            "join_column": "midi_filename",
            "where": {},
            "sample": None,
        }
    return {
        "configured": True,
        "dataset": cfg.io.dataset,
        "metadata_csv": str(metadata_path) if metadata_path is not None else None,
        "metadata_sha256": _metadata_sha256(metadata_path),
        "join_column": cfg.io.join_column,
        "where": dict(cfg.io.where),
        "sample": cfg.io.sample.model_dump() if cfg.io.sample is not None else None,
    }


def _by_value(
    stems: Mapping[Path, str],
    metadata: dict[str, dict[str, str]] | None,
    where: Mapping[str, list[str]],
) -> dict[str, dict[str, int]]:
    if not where or metadata is None:
        return {}
    by_value: dict[str, dict[str, int]] = {}
    for column in sorted(where):
        per_value: dict[str, int] = {}
        for stem in stems.values():
            row = metadata.get(stem)
            if row is None:
                continue
            value = row.get(column)
            if not value:
                continue
            per_value[value] = per_value.get(value, 0) + 1
        by_value[column] = {value: per_value[value] for value in sorted(per_value)}
    return by_value


def _counts_text(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _metadata_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_sha256(units: Sequence[Path], unit_root: Path) -> str:
    relative: list[str] = []
    for unit in units:
        try:
            rendered = unit.relative_to(unit_root).as_posix()
        except ValueError:
            rendered = unit.as_posix()
        relative.append(rendered)
    return hashlib.sha256("\n".join(sorted(relative)).encode("utf-8")).hexdigest()


def select_references(
    cfg: PipelineConfig,
    references: Sequence[Path],
    *,
    apply_sample: bool = True,
) -> SelectionResult:
    filtered = _filter_references(cfg, references)
    excluded_by_where = (
        len(filtered.refs_all)
        - len(filtered.unmatched)
        - len(filtered.refs_selected)
    )
    if apply_sample:
        units = sample_units(filtered.refs_selected, cfg.io.sample)
    else:
        units = list(filtered.refs_selected)

    counts = {
        "discovered": len(filtered.refs_all),
        "unmatched": len(filtered.unmatched),
        "excluded_by_where": excluded_by_where,
        "unpaired_audio": 0,
        "selected_before_sample": len(filtered.refs_selected),
        "selected": len(units),
    }
    if not units:
        raise SelectionError(f"selection matched no files ({_counts_text(counts)})")

    provenance_inputs = _selection_provenance(cfg, filtered.metadata_path)
    logger.debug(
        "selection resolved %d/%d reference files (%d unmatched)",
        counts["selected"],
        counts["discovered"],
        counts["unmatched"],
    )
    return SelectionResult(
        units=units,
        references=list(units),
        unit_kind="reference_midi",
        counts=counts,
        by_value=_by_value(
            {unit: unit.stem for unit in units},
            filtered.metadata,
            provenance_inputs["where"],
        ),
        unmatched=filtered.unmatched,
        unpaired_audio=[],
        notice=split_notice(cfg, len(filtered.refs_all)),
        **provenance_inputs,
    )


def select_audio(
    cfg: PipelineConfig,
    audio: Sequence[Path],
    references: Sequence[Path],
    *,
    unit_kind: Literal["recording", "audio"],
    apply_sample: bool = True,
    always_pair: bool = False,
) -> SelectionResult:
    audio_all = list(audio)
    references_all = list(references)
    where: dict[str, list[str]] = dict(cfg.io.where)

    if (
        (where or always_pair)
        and unit_kind == "audio"
        and not cfg.io.file_naming.startswith("{stem}")
    ):
        raise SelectionError(
            "selection needs io.file_naming to start with '{stem}' to pair rendered "
            f"audio (got '{cfg.io.file_naming}')"
        )

    metadata: dict[str, dict[str, str]] | None
    metadata_path: Path | None
    if where:
        filtered = _filter_references(cfg, references_all)
        pairing = pair_audio_to_reference(audio_all, filtered.refs_all)
        selected_references = set(filtered.refs_selected)
        audio_to_reference = {
            item: reference
            for item, reference in pairing.mapping.items()
            if reference in selected_references
        }
        kept = list(audio_to_reference)
        unpaired = list(pairing.unpaired_audio)
        unmatched = filtered.unmatched
        excluded_by_where = (
            len(filtered.refs_all)
            - len(filtered.unmatched)
            - len(filtered.refs_selected)
        )
        metadata = filtered.metadata
        metadata_path = filtered.metadata_path
    elif always_pair:
        # Pair every recording against the full reference list and drop the
        # unpaired ones; used by benchmark audio mode so a sampled recording
        # always has a reference.
        pairing = pair_audio_to_reference(audio_all, references_all)
        audio_to_reference = dict(pairing.mapping)
        kept = list(audio_to_reference)
        unpaired = list(pairing.unpaired_audio)
        unmatched = []
        excluded_by_where = 0
        metadata = None
        metadata_path = None
    else:
        audio_to_reference = {}
        kept = list(audio_all)
        unpaired = []
        unmatched = []
        excluded_by_where = 0
        metadata = None
        metadata_path = None

    if apply_sample:
        units = sample_units(kept, cfg.io.sample)
    else:
        units = list(kept)

    if where or always_pair:
        references_out = sorted({audio_to_reference[unit] for unit in units})
        stems = {unit: audio_to_reference[unit].stem for unit in units}
    else:
        references_out = list(references_all)
        stems = {}

    counts = {
        "discovered": len(audio_all),
        "unmatched": len(unmatched),
        "excluded_by_where": excluded_by_where,
        "unpaired_audio": len(unpaired),
        "selected_before_sample": len(kept),
        "selected": len(units),
    }
    if not units:
        raise SelectionError(f"selection matched no files ({_counts_text(counts)})")

    provenance_inputs = _selection_provenance(cfg, metadata_path)
    logger.debug(
        "selection resolved %d/%d audio files (%d unpaired)",
        counts["selected"],
        counts["discovered"],
        counts["unpaired_audio"],
    )
    return SelectionResult(
        units=units,
        references=references_out,
        unit_kind=unit_kind,
        counts=counts,
        by_value=_by_value(stems, metadata, provenance_inputs["where"]),
        unmatched=unmatched,
        unpaired_audio=unpaired,
        notice=split_notice(cfg, len(audio_all)),
        **provenance_inputs,
    )


def split_notice(cfg: PipelineConfig, n_files: int) -> str | None:
    if cfg.io.where:
        return None
    if cfg.io.dataset is None:
        return None
    metadata_dir = Path(cfg.io.corpus_root) / cfg.io.dataset / "metadata"
    if not metadata_dir.is_dir():
        return None
    labelled: list[str] = []
    for csv_path in sorted(metadata_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if "split" in header:
            labelled.append(csv_path.name)
    if not labelled:
        return None
    listing = ", ".join(f"{name}: split" for name in labelled)
    return (
        f"note: {cfg.io.dataset} metadata labels splits ({listing}); "
        f"no selection set, using all {n_files} files"
    )
