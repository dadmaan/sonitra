from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "export_model_baselines.py"

BEGIN = "<!-- BEGIN GENERATED: baselines -->"
END = "<!-- END GENERATED: baselines -->"

PROSE_BEFORE = "# Model cards\n\nIntro prose that must survive.\n\n### Measured baselines\n\n"
PROSE_AFTER = "\n\n---\n[← Back to README](../README.md)\n"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_model_baselines", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def emb() -> ModuleType:
    return _load_module()


def _baseline_row(
    transcriber: str = "basic_pitch",
    n: int = 100,
    *,
    n_succeeded: int | None = None,
    condition: str = "baseline",
    **metrics: float,
) -> dict:
    row: dict = {
        "condition": condition,
        "transcriber": transcriber,
        "n_files": n,
        "n_succeeded": n if n_succeeded is None else n_succeeded,
        "overrides": {},
        "note.onset_f1": 0.500,
        "note.onset_offset_f1": 0.400,
        "note.onset_offset_velocity_f1": 0.300,
        "expressive.velocity_corr": 0.200,
        "frame.f1": 0.600,
    }
    row.update(metrics)
    return row


def _write_run(
    corpus_root: Path,
    dataset: str,
    run: str,
    rows: list[dict],
    *,
    input_type: str | None = "midi",
    write_config: bool = True,
) -> Path:
    run_dir = corpus_root / dataset / "benchmark" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"summary": rows, "degradation": {}, "timing": {"overall_seconds": 1.0}}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if write_config:
        render: dict = {}
        if input_type is not None:
            render["input_type"] = input_type
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump({"render_pipeline": render}), encoding="utf-8"
        )
    return run_dir


def _write_doc(doc: Path, *, begin: str = BEGIN, end: str = END, extra_begin: bool = False) -> Path:
    doc.parent.mkdir(parents=True, exist_ok=True)
    body = PROSE_BEFORE + begin + "\n" + end
    if extra_begin:
        body += "\n\n" + begin + "\n" + end
    doc.write_text(body + PROSE_AFTER, encoding="utf-8")
    return doc


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    return root


@pytest.fixture()
def doc(tmp_path: Path) -> Path:
    return _write_doc(tmp_path / "docs" / "model-cards.md")


def _run(emb: ModuleType, corpus_root: Path, doc: Path, *extra: str) -> int:
    return emb.main(["--corpus-root", str(corpus_root), "--doc", str(doc), *extra])


def _block(doc: Path) -> str:
    text = doc.read_text(encoding="utf-8")
    return text.split(BEGIN, 1)[1].split(END, 1)[0]


# --- selection -------------------------------------------------------------


def test_discovers_baseline_rows_across_datasets(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row(n=1276)])
    _write_run(corpus_root, "guitarset", "guitar_only_MIDI", [_baseline_row(n=360)])

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "maestro-v3" in block
    assert "guitarset" in block
    assert "1276" in block
    assert "360" in block


def test_ignores_non_baseline_conditions(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    rows = [
        _baseline_row(n=100, **{"note.onset_f1": 0.111}),
        _baseline_row(n=100, condition="baseline_x", **{"note.onset_f1": 0.999}),
        _baseline_row(n=100, condition="inst=piano_reverb=off", **{"note.onset_f1": 0.888}),
    ]
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", rows)

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "0.111" in block
    assert "0.999" not in block
    assert "0.888" not in block


def test_excludes_non_allowlisted_runs(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row(**{"note.onset_f1": 0.111})])
    _write_run(
        corpus_root,
        "maestro-v3",
        "vintage_scenarios_MIDI",
        [_baseline_row(**{"note.onset_f1": 0.999})],
    )

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "0.111" in block
    assert "0.999" not in block


def test_does_not_dedupe_runs_differing_by_input_type(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(
        corpus_root,
        "maestro-v3",
        "piano_only_AUDIO",
        [_baseline_row(**{"note.onset_f1": 0.662})],
        input_type="audio",
    )
    _write_run(
        corpus_root,
        "maestro-v3",
        "piano_only_MIDI",
        [_baseline_row(**{"note.onset_f1": 0.817})],
        input_type="midi",
    )

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "0.662" in block
    assert "0.817" in block


def test_input_type_comes_from_config_not_directory_name(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    # Directory says AUDIO, config says midi. The config wins.
    _write_run(
        corpus_root,
        "maestro-v3",
        "piano_only_AUDIO",
        [_baseline_row()],
        input_type="midi",
    )

    assert _run(emb, corpus_root, doc) == 0

    row_lines = [ln for ln in _block(doc).splitlines() if "maestro-v3" in ln]
    assert len(row_lines) == 1
    assert "midi" in row_lines[0]
    assert "audio" not in row_lines[0]


def test_run_without_config_is_skipped_with_warning(
    emb: ModuleType, corpus_root: Path, doc: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row(**{"note.onset_f1": 0.111})])
    _write_run(
        corpus_root,
        "guitarset",
        "guitar_only_MIDI",
        [_baseline_row(**{"note.onset_f1": 0.999})],
        write_config=False,
    )

    assert _run(emb, corpus_root, doc) == 0

    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "guitar_only_MIDI" in captured.err

    block = _block(doc)
    assert "0.111" in block
    assert "0.999" not in block


def test_run_without_input_type_field_is_skipped(
    emb: ModuleType, corpus_root: Path, doc: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()], input_type=None)

    assert _run(emb, corpus_root, doc) == 0

    assert "warning:" in capsys.readouterr().err
    assert "basic_pitch" not in _block(doc)


def test_non_allowlisted_runs_without_config_are_summarised_not_listed(
    emb: ModuleType, corpus_root: Path, doc: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    _write_run(corpus_root, "test", "reverb_sweep", [_baseline_row()], write_config=False)
    _write_run(corpus_root, "test", "venue_scenarios", [_baseline_row()], write_config=False)

    assert _run(emb, corpus_root, doc) == 0

    err = capsys.readouterr().err
    assert "skipped 2 non-allowlisted run(s)" in err
    assert "reverb_sweep" not in err
    assert "venue_scenarios" not in err


def test_colliding_row_keys_error(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    # Same dataset, same input_type, same transcriber, two allowlisted runs.
    _write_run(corpus_root, "maestro-v3", "piano_only_a", [_baseline_row()], input_type="midi")
    _write_run(corpus_root, "maestro-v3", "piano_only_b", [_baseline_row()], input_type="midi")

    before = doc.read_text(encoding="utf-8")
    assert _run(emb, corpus_root, doc) == 1
    assert doc.read_text(encoding="utf-8") == before


# --- omission note ---------------------------------------------------------


def test_transcriber_only_in_excluded_runs_is_omitted_with_note(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row("basic_pitch")])
    _write_run(corpus_root, "test", "transkun_baseline", [_baseline_row("transkun", n=1)])

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    table_rows = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
    assert not any("transkun" in ln for ln in table_rows)
    assert "transkun" in block  # named in the omission note
    assert block.lower().count("transkun") >= 1


def test_transcriber_with_qualifying_row_gets_no_omission_note(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row("transkun")])
    _write_run(corpus_root, "test", "transkun_baseline", [_baseline_row("transkun", n=1)])

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "omitted" not in block.lower()


# --- rendering -------------------------------------------------------------


def test_nan_metric_renders_as_em_dash(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    row = _baseline_row()
    row["expressive.velocity_corr"] = float("nan")
    _write_run(corpus_root, "guitarset", "guitar_only_MIDI", [row])

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "—" in block
    assert "nan" not in block.lower()


def test_absent_metric_renders_as_em_dash(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    row = _baseline_row()
    del row["expressive.velocity_corr"]
    _write_run(corpus_root, "guitarset", "guitar_only_MIDI", [row])

    assert _run(emb, corpus_root, doc) == 0
    assert "—" in _block(doc)


def test_partial_success_renders_succeeded_over_total(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(
        corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row(n=10, n_succeeded=8)]
    )

    assert _run(emb, corpus_root, doc) == 0
    assert "8/10" in _block(doc)


def test_full_success_renders_bare_count(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row(n=10)])

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "10/10" not in block
    assert "| 10 |" in block


def test_rows_sorted_deterministically(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()], input_type="midi")
    _write_run(corpus_root, "bsed", "piano_only_AUDIO", [_baseline_row()], input_type="audio")
    _write_run(corpus_root, "guitarset", "guitar_only_MIDI", [_baseline_row()], input_type="midi")

    assert _run(emb, corpus_root, doc) == 0

    datasets = [
        ln.split("|")[1].strip()
        for ln in _block(doc).splitlines()
        if ln.strip().startswith("|") and "---" not in ln
    ][1:]
    assert datasets == sorted(datasets)


# --- injection -------------------------------------------------------------


def test_injection_preserves_surrounding_prose(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])

    assert _run(emb, corpus_root, doc) == 0

    text = doc.read_text(encoding="utf-8")
    assert text.startswith(PROSE_BEFORE)
    assert text.endswith(PROSE_AFTER)
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1


def test_missing_markers_errors_and_leaves_file_untouched(
    emb: ModuleType, corpus_root: Path, tmp_path: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    doc = tmp_path / "docs" / "model-cards.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# Model cards\n\nNo markers here.\n", encoding="utf-8")
    before = doc.read_text(encoding="utf-8")

    assert _run(emb, corpus_root, doc) == 1
    assert doc.read_text(encoding="utf-8") == before


def test_duplicate_begin_marker_errors(
    emb: ModuleType, corpus_root: Path, tmp_path: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    doc = _write_doc(tmp_path / "docs" / "model-cards.md", extra_begin=True)
    before = doc.read_text(encoding="utf-8")

    assert _run(emb, corpus_root, doc) == 1
    assert doc.read_text(encoding="utf-8") == before


def test_rerun_is_idempotent(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])

    assert _run(emb, corpus_root, doc) == 0
    first = doc.read_text(encoding="utf-8")
    assert _run(emb, corpus_root, doc) == 0
    assert doc.read_text(encoding="utf-8") == first


def test_provenance_date_is_max_summary_mtime_not_today(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    older = _write_run(corpus_root, "bsed", "piano_only_MIDI", [_baseline_row()])
    newer = _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    # 2021-03-04 and 2022-05-06 UTC
    os.utime(older / "summary.json", (1614816000, 1614816000))
    os.utime(newer / "summary.json", (1651795200, 1651795200))

    assert _run(emb, corpus_root, doc) == 0

    block = _block(doc)
    assert "2022-05-06" in block
    assert "2021-03-04" not in block


def test_check_returns_zero_when_in_sync(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])

    assert _run(emb, corpus_root, doc) == 0
    written = doc.read_text(encoding="utf-8")

    assert _run(emb, corpus_root, doc, "--check") == 0
    assert doc.read_text(encoding="utf-8") == written


def test_check_returns_nonzero_when_stale(emb: ModuleType, corpus_root: Path, doc: Path) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    assert _run(emb, corpus_root, doc) == 0
    written = doc.read_text(encoding="utf-8")

    _write_run(corpus_root, "guitarset", "guitar_only_MIDI", [_baseline_row()])

    assert _run(emb, corpus_root, doc, "--check") != 0
    assert doc.read_text(encoding="utf-8") == written


def test_stdout_prints_block_and_leaves_doc_untouched(
    emb: ModuleType, corpus_root: Path, doc: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    before = doc.read_text(encoding="utf-8")

    assert _run(emb, corpus_root, doc, "--stdout") == 0

    captured = capsys.readouterr()
    assert BEGIN in captured.out
    assert "maestro-v3" in captured.out
    assert doc.read_text(encoding="utf-8") == before


def test_check_and_stdout_are_mutually_exclusive(
    emb: ModuleType, corpus_root: Path, doc: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])

    with pytest.raises(SystemExit):
        _run(emb, corpus_root, doc, "--check", "--stdout")


def test_refuses_doc_path_inside_corpus_root(
    emb: ModuleType, corpus_root: Path
) -> None:
    _write_run(corpus_root, "maestro-v3", "piano_only_MIDI", [_baseline_row()])
    doc = _write_doc(corpus_root / "maestro-v3" / "model-cards.md")
    before = doc.read_text(encoding="utf-8")

    assert _run(emb, corpus_root, doc) == 1
    assert doc.read_text(encoding="utf-8") == before
