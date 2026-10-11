"""Checks that the annotated reference config documents and honours its own knobs."""

from __future__ import annotations

import re
from pathlib import Path

from sonitra.api.app import create_app
from sonitra.config import load_config
from sonitra.effects import TuningOffsetConfig
from sonitra.pipeline import run_pipeline
from sonitra.transcribe.configs import HftTransformerTranscriberConfig

from tests.presets._presets import CONFIG_DIR

# The annotated reference, addressed directly: default_config_path() resolves the
# same file, but resolving it here keeps this suite's subject explicit.
SOURCE_PATH: Path = CONFIG_DIR / "source.yaml"

# Rendering several fixtures at once is what shows the reference is runnable as it
# stands rather than loadable only; a single file would pass on a config that
# broke the second one.
_MIN_RENDERABLE_FIXTURES = 2

# The base fields every transcriber carries. They are not knobs of the hFT entry
# itself, so documenting them there is not required.
_TRANSCRIBER_BASE_FIELDS = frozenset({"enabled", "name", "type"})


def _documented_fields(body: str, fields: set[str]) -> set[str]:
    """Fields among ``fields`` that ``body`` carries as a commented ``key:`` line."""
    return {
        field
        for field in fields
        if re.search(rf"^[ \t]*#[ \t]+{field}:", body, re.M)
    }


def test_source_yaml_renders_the_fixtures(corpus_dir: Path, tmp_path: Path) -> None:
    """The shipped reference renders real files, and what it renders is not clipped.

    A reference a reader copies must survive a run on its own, and its defaults
    must leave the output below the clip threshold it also ships, so a copied
    config produces clean audio rather than something the quality gate flags.
    """
    midi_files = sorted(corpus_dir.glob("*.mid"))
    assert len(midi_files) >= _MIN_RENDERABLE_FIXTURES, (
        f"{corpus_dir} holds fewer than {_MIN_RENDERABLE_FIXTURES} midi fixtures"
    )

    cfg = load_config(SOURCE_PATH)
    result = run_pipeline(midi_files, tmp_path, config=cfg)
    assert result.succeeded >= _MIN_RENDERABLE_FIXTURES

    succeeded = [entry for entry in result.log if entry["status"] == "succeeded"]
    assert len(succeeded) == result.succeeded
    for entry in succeeded:
        flags = entry["quality_flags"]
        assert flags["is_clipped"] is False, entry
        assert flags["peak"] < cfg.quality_gates.clip_threshold, entry


def test_source_yaml_documents_the_io_filter_keys() -> None:
    """Every file-selection key is documented, and no live selection is shipped.

    A shipped ``where``/``sample`` would silently shrink a reader's first run to
    the reference's own subset, so the keys are shown commented and the loaded
    values stay empty.
    """
    text = SOURCE_PATH.read_text()
    cfg = load_config(SOURCE_PATH)
    assert cfg.io.metadata_csv is None
    assert cfg.io.where == {}
    assert cfg.io.sample is None
    assert not re.search(r"^selection:", text, re.M)
    for key in ("metadata_csv", "join_column", "where", "sample", "n", "seed"):
        assert re.search(rf"^[#\s]+{key}:", text, re.M), key
    for key in ("metadata_csv", "join_column", "where", "sample"):
        assert re.search(rf"^\s+#?\s*{key}:", text, re.M), key


def test_source_yaml_documents_the_tuning_offset_knobs() -> None:
    """The commented tuning-offset entry names every knob the effect accepts.

    The set comes from the model, so a knob added later has to be documented
    here or this fails with its name rather than passing on a stale hand list.
    """
    text = SOURCE_PATH.read_text()
    entry = re.search(
        r"^[ \t]*# - type: TuningOffset$(?P<body>(?:\n[ \t]*#.*)*)", text, re.M
    )
    assert entry, f"{SOURCE_PATH} has no commented TuningOffset entry"
    body = entry.group("body")

    expected = {"cents"} | (set(TuningOffsetConfig.model_fields) - {"type"})
    missing = expected - _documented_fields(body, expected)
    assert not missing, f"undocumented TuningOffset knobs: {sorted(missing)}"
    # The engine choice is the reason the offset is not a native plugin.
    assert "time_stretch" in body


def test_source_yaml_documents_the_hft_transformer_knobs() -> None:
    """The commented hFT entry names every knob of its transcriber config.

    The set comes from the model, so a knob added later has to be documented
    here or this fails with its name rather than passing on a stale hand list.
    """
    text = SOURCE_PATH.read_text()
    entry = re.search(
        r"^[ \t]*# - type: hft_transformer$(?P<body>(?:\n[ \t]*#.*)*)", text, re.M
    )
    assert entry, f"{SOURCE_PATH} has no commented hft_transformer entry"
    body = entry.group("body")

    expected = set(HftTransformerTranscriberConfig.model_fields) - set(
        _TRANSCRIBER_BASE_FIELDS
    )
    missing = expected - _documented_fields(body, expected)
    assert not missing, f"undocumented hft_transformer knobs: {sorted(missing)}"
    # The two facts that make the entry usable without reading the code.
    assert "GPU:N" in body
    assert "converted" in body
    assert "time-axis" in body


def test_source_yaml_names_the_command_each_worker_key_belongs_to() -> None:
    """The reference must say which command each worker key applies to.

    Two identically named keys sit under different sections, and only one of
    them affects a benchmark, so each comment names its own command.
    """
    text = SOURCE_PATH.read_text()
    expected = {
        "transcription": "sonitra transcribe",
        "evaluation": "sonitra evaluate",
    }

    for section, command in expected.items():
        # Locate the key by scanning the section's body, since line numbers drift.
        body = re.search(rf"^{section}:\n(?P<lines>(?:[ \t]+.*\n)+)", text, re.M)
        assert body, f"{SOURCE_PATH} has no {section} section"
        match = re.search(r"^[ \t]+max_workers:.*$", body.group("lines"), re.M)
        assert match, f"{SOURCE_PATH} {section} section documents no max_workers"
        assert command in match.group(0), match.group(0)


def test_source_yaml_mentions_the_basic_pitch_knobs() -> None:
    """The reference documents the basic-pitch knobs that change its output."""
    text = SOURCE_PATH.read_text()
    for knob in ("melodia_trick", "multiple_pitch_bends", "save_raw_outputs"):
        assert re.search(rf"^[ \t]*{knob}:", text, re.M), knob


def test_api_default_config_is_the_shipped_reference() -> None:
    """An app built without a config serves the shipped reference verbatim."""
    assert create_app().state.config == load_config(SOURCE_PATH)