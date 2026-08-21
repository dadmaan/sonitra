from __future__ import annotations

import csv
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_mixed_effects_analysis.py"
R_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "mixed_effects_analysis.R"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_mixed_effects_analysis", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types via sys.modules[cls.__module__], so a
    # module loaded by path has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mea() -> ModuleType:
    return _load_module()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "condition": "baseline",
        "transcriber": "basic_pitch",
        "song": "song_a",
        "status": "succeeded",
        "note.onset_f1": 0.8,
        "meta.canonical_composer": "Bach",
        "meta.duration": 300.0,
        "meta.year": 2011,
    }
    row.update(overrides)
    return row


def _balanced_rows(
    n_songs: int = 12,
    conditions: tuple[str, ...] = ("baseline", "shellac_bandlimit_mild"),
    *,
    noise: float = 0.02,
) -> list[dict[str, object]]:
    """A crossed song x condition design resembling a real benchmark table.

    Seeded, so the fit is reproducible. The residual noise matters: a response
    that is an exact deterministic function of the predictors gives glmmTMB a
    singular (non-positive-definite) Hessian and no likelihood.
    """
    rng = random.Random(20260821)
    rows: list[dict[str, object]] = []
    for song_index in range(n_songs):
        song_effect = rng.gauss(0.0, 0.05)
        for condition_index, condition in enumerate(conditions):
            value = 0.82 - 0.06 * condition_index + song_effect + rng.gauss(0.0, noise)
            rows.append(
                _row(
                    condition=condition,
                    song=f"song_{song_index:03d}",
                    **{
                        # Kept strictly inside (0, 1) -- beta_family's support.
                        "note.onset_f1": round(min(max(value, 0.05), 0.95), 6),
                        "meta.canonical_composer": f"composer_{song_index % 3}",
                        "meta.duration": 120.0 + 30.0 * song_index,
                        "meta.year": 2004 + (song_index % 8),
                    },
                )
            )
    return rows


# --------------------------------------------------------------------------
# column validation
# --------------------------------------------------------------------------


def test_missing_columns_reports_absent_model_columns(mea: ModuleType) -> None:
    header = ["condition", "song", "note.onset_f1"]
    missing = mea.missing_columns(header)
    assert "meta.canonical_composer" in missing
    assert "meta.duration" in missing
    assert "meta.year" in missing
    assert "condition" not in missing


def test_missing_columns_empty_when_all_present(mea: ModuleType) -> None:
    assert mea.missing_columns(list(_row())) == []


# --------------------------------------------------------------------------
# input summary
# --------------------------------------------------------------------------


def test_summarise_input_counts_design(mea: ModuleType, tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "table.csv", _balanced_rows(n_songs=6))
    summary = mea.summarise_input(path)
    assert summary.n_rows == 12
    assert summary.n_songs == 6
    assert summary.n_composers == 3
    assert summary.conditions == ["baseline", "shellac_bandlimit_mild"]
    assert summary.transcribers == ["basic_pitch"]
    assert summary.n_boundary == 0
    assert summary.n_incomplete == 0
    assert summary.n_not_succeeded == 0


def test_summarise_input_flags_beta_boundary_values(mea: ModuleType, tmp_path: Path) -> None:
    rows = [
        _row(song="a", **{"note.onset_f1": 0.0}),
        _row(song="b", **{"note.onset_f1": 1.0}),
        _row(song="c", **{"note.onset_f1": 0.5}),
    ]
    path = _write_csv(tmp_path / "table.csv", rows)
    assert mea.summarise_input(path).n_boundary == 2


def test_summarise_input_counts_incomplete_and_failed_rows(mea: ModuleType, tmp_path: Path) -> None:
    rows = [
        _row(song="a"),
        _row(song="b", **{"note.onset_f1": ""}),
        _row(song="c", **{"meta.duration": ""}),
        _row(song="d", status="failed"),
    ]
    path = _write_csv(tmp_path / "table.csv", rows)
    summary = mea.summarise_input(path)
    assert summary.n_incomplete == 2
    assert summary.n_not_succeeded == 1


def test_summarise_input_detects_multiple_transcribers(mea: ModuleType, tmp_path: Path) -> None:
    rows = [_row(song="a"), _row(song="b", transcriber="onsets_and_frames")]
    path = _write_csv(tmp_path / "table.csv", rows)
    assert mea.summarise_input(path).transcribers == ["basic_pitch", "onsets_and_frames"]


def test_summarise_input_tolerates_missing_optional_columns(mea: ModuleType, tmp_path: Path) -> None:
    """`transcriber`/`status` are advisory -- absent ones must not raise."""
    rows = []
    for row in _balanced_rows(n_songs=4):
        row.pop("transcriber")
        row.pop("status")
        rows.append(row)
    path = _write_csv(tmp_path / "table.csv", rows)
    summary = mea.summarise_input(path)
    assert summary.transcribers == []
    assert summary.n_not_succeeded == 0


# --------------------------------------------------------------------------
# paths and command construction
# --------------------------------------------------------------------------


def test_default_output_dir_sits_next_to_input(mea: ModuleType, tmp_path: Path) -> None:
    csv_path = tmp_path / "run" / "regression_table_with_metadata.csv"
    assert mea.default_output_dir(csv_path) == tmp_path / "run" / "regression_analysis"


def test_build_command_passes_input_output_and_ref_level(mea: ModuleType, tmp_path: Path) -> None:
    command = mea.build_command(
        rscript="/usr/bin/Rscript",
        r_script=R_SCRIPT_PATH,
        csv_path=tmp_path / "in.csv",
        output_dir=tmp_path / "out",
        ref_level="baseline",
    )
    assert command[0] == "/usr/bin/Rscript"
    assert str(R_SCRIPT_PATH) in command
    assert "--ref-level" in command
    assert command[command.index("--ref-level") + 1] == "baseline"
    assert str(tmp_path / "in.csv") in command
    assert str(tmp_path / "out") in command


def test_find_rscript_prefers_explicit_path(mea: ModuleType, tmp_path: Path) -> None:
    fake = tmp_path / "Rscript"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    assert mea.find_rscript(str(fake), env={}) == str(fake)


def test_find_rscript_falls_back_to_env_var(mea: ModuleType, tmp_path: Path) -> None:
    fake = tmp_path / "Rscript"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    assert mea.find_rscript(None, env={"SONITRA_RSCRIPT": str(fake)}) == str(fake)


def test_find_rscript_returns_none_when_absent(mea: ModuleType, tmp_path: Path) -> None:
    assert mea.find_rscript(str(tmp_path / "nope"), env={}) is None


# --------------------------------------------------------------------------
# main() failure paths (no R required)
# --------------------------------------------------------------------------


def test_main_errors_when_input_missing(mea: ModuleType, tmp_path: Path, capsys) -> None:
    code = mea.main(["--input", str(tmp_path / "absent.csv")])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_main_errors_on_missing_model_columns(mea: ModuleType, tmp_path: Path, capsys) -> None:
    path = _write_csv(tmp_path / "table.csv", [{"condition": "baseline", "song": "a"}])
    code = mea.main(["--input", str(path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "note.onset_f1" in err


def test_main_errors_with_install_hint_when_rscript_absent(
    mea: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    path = _write_csv(tmp_path / "table.csv", _balanced_rows(n_songs=4))
    monkeypatch.setattr(mea, "find_rscript", lambda *a, **k: None)
    code = mea.main(["--input", str(path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "Rscript" in err
    assert "glmmtmb" in err.lower()


def test_main_dry_run_reports_without_invoking_r(
    mea: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    path = _write_csv(tmp_path / "table.csv", _balanced_rows(n_songs=4))

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("R must not be invoked for --dry-run")

    monkeypatch.setattr(mea.subprocess, "run", _explode)
    code = mea.main(["--input", str(path), "--dry-run"])
    assert code == 0
    assert "8 rows" in capsys.readouterr().out


# --------------------------------------------------------------------------
# end-to-end against a real R + glmmTMB install
# --------------------------------------------------------------------------


def _rscript_or_skip() -> str:
    module = _load_module()
    import os

    rscript = module.find_rscript(None, env=dict(os.environ))
    if rscript is None:
        pytest.skip("Rscript not available")
    if module.missing_r_packages(rscript):
        pytest.skip("R present but glmmTMB/jsonlite missing")
    return rscript


@pytest.mark.requires_r
@pytest.mark.slow
def test_end_to_end_fit_writes_all_artifacts(mea: ModuleType, tmp_path: Path) -> None:
    rscript = _rscript_or_skip()
    path = _write_csv(tmp_path / "table.csv", _balanced_rows(n_songs=30))
    out = tmp_path / "regression_analysis"

    code = mea.main(["--input", str(path), "--rscript", rscript])
    assert code == 0

    for name in (
        "model_summary.txt",
        "fixed_effects.csv",
        "random_effects_composer.csv",
        "random_effects_song.csv",
        "model_meta.json",
        "fit.R",
    ):
        assert (out / name).exists(), f"missing artifact: {name}"

    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["n_obs"] == 60
    assert meta["family"] == "beta"
    assert meta["ref_level"] == "baseline"
    assert meta["formula"].startswith("note.onset_f1 ~")
    assert meta["input_sha256"]
    assert meta["r_version"]
    assert meta["packages"]["glmmTMB"]

    terms = [row["term"] for row in csv.DictReader((out / "fixed_effects.csv").open())]
    assert "(Intercept)" in terms
    assert "conditionshellac_bandlimit_mild" in terms

    # fit.R must be the byte-identical script that produced the results.
    assert (out / "fit.R").read_bytes() == R_SCRIPT_PATH.read_bytes()


@pytest.mark.requires_r
@pytest.mark.slow
def test_end_to_end_errors_on_unknown_ref_level(mea: ModuleType, tmp_path: Path, capsys) -> None:
    rscript = _rscript_or_skip()
    path = _write_csv(tmp_path / "table.csv", _balanced_rows(n_songs=8))
    code = mea.main(["--input", str(path), "--rscript", rscript, "--ref-level", "not_a_condition"])
    assert code != 0
    assert "not_a_condition" in capsys.readouterr().err


def test_r_script_is_shipped_and_executable() -> None:
    assert R_SCRIPT_PATH.exists()
    assert shutil.which("Rscript") is None or R_SCRIPT_PATH.stat().st_size > 0


@pytest.mark.requires_r
@pytest.mark.slow
def test_end_to_end_survives_a_degenerate_fit(mea: ModuleType, tmp_path: Path, capsys) -> None:
    """A noiseless response gives a singular Hessian and no AIC -- report, don't crash."""
    rscript = _rscript_or_skip()
    path = _write_csv(tmp_path / "table.csv", _balanced_rows(n_songs=8, noise=0.0))
    code = mea.main(["--input", str(path), "--rscript", rscript])
    assert code == 0
    captured = capsys.readouterr()
    meta = json.loads((tmp_path / "regression_analysis" / "model_meta.json").read_text())
    if not (meta["converged"] and meta["positive_definite_hessian"]):
        assert "did not converge" in captured.err


def test_r_locale_env_prefers_c_utf8(mea: ModuleType) -> None:
    env = mea.r_locale_env({"LANG": "en_US.UTF-8"}, {"c", "c.utf8", "posix"})
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["LANG"] == "C.UTF-8"


def test_r_locale_env_pins_only_numerics_without_c_utf8(mea: ModuleType) -> None:
    """Falling back to plain C would make R escape non-ASCII composer names."""
    env = mea.r_locale_env({"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}, {"c", "posix"})
    assert "LC_ALL" not in env
    assert env["LC_NUMERIC"] == "C"
    assert env["LANG"] == "en_US.UTF-8"
