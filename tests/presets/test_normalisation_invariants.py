"""Invariants the shipped normalisation and quality-gate settings must satisfy.

Every rule is derived from the config models and the preset files themselves, so
a new preset or a newly added field is covered without editing the rule. The
rules guard the two ways normalisation can silently undo a preset's intent:
a positive gain that output levelling cancels, and a level-dependent effect left
after the chain while levelling runs only at the output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sonitra.benchmark.conditions import apply_overrides, expand_conditions
from sonitra.config import (
    NormalisationSection,
    NormalisationStage,
    PipelineConfig,
    QualityGatesSection,
    load_config,
)
from sonitra.effects import ClippingConfig
from tests.presets._presets import CONFIG_DIR, PAPER_DIR, paper_presets, presets

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = CONFIG_DIR / "source.yaml"
_SRC_DIR = _REPO_ROOT / "src"

# Effects whose output character depends on the input level. They need a known
# input level to behave the same way on every file, which is what an input stage
# provides and what output-only levelling cannot.
_LEVEL_DEPENDENT_TYPES = frozenset({"Distortion", "Compressor", "Limiter", "Clipping"})

# Presets that run an enabled level-dependent effect while levelling only after
# the chain. Each is a documented per-study level decision rather than an
# oversight, so this is the open to-do: it should only ever shrink, and a file
# that stops offending must be removed from the set.
_ALLOWED_POST_PLACEMENT = frozenset(
    {
        "benchmark/methods/numerics_check.yaml",
        "benchmark/smoke/benchmark_test.yaml",
        "benchmark/smoke/gaps_test.yaml",
        "benchmark/smoke/guitarset_test.yaml",
        "benchmark/smoke/hft_smoke.yaml",
        "benchmark/smoke/musicnet_test.yaml",
        "benchmark/telephone_channel/telephone_scenarios.yaml",
        "benchmark/transkun/transkun_baseline.yaml",
        "benchmark/transkun/transkun_baseline_batch16_strict.yaml",
        "benchmark/transkun/transkun_baseline_gpu.yaml",
        "benchmark/transkun/transkun_baseline_gpu_batch8_strict.yaml",
        "examples/dawdreamer_vital_delayed_flight_pedalboard.yaml",
        "examples/dawdreamer_vital_goodies_pedalboard.yaml",
        "examples/dawdreamer_vital_pedalboard.yaml",
        "examples/pedalboard_all_effects.yaml",
        "examples/pedalboard_baseline.yaml",
        "examples/pedalboard_parallel.yaml",
    }
)


def _resolved_conditions(cfg: PipelineConfig) -> list[tuple[str, PipelineConfig]]:
    """Every benchmark condition of ``cfg`` as a resolved ``(name, config)`` pair."""
    return [
        (condition.name, apply_overrides(cfg, condition.overrides))
        for condition in expand_conditions(cfg.benchmark)
    ]


def _positive_gain_offenders(cfg: PipelineConfig, label: str) -> list[str]:
    """Describe enabled positive gains no later effect can pin the level for.

    The rule only applies where a stage levels the input before the effects, so
    each effect sees a known level. Under output-only levelling the guard simply
    cancels the boost, so a positive gain there is not flagged.
    """
    if cfg.normalisation.placement not in ("pre", "both"):
        return []
    offenders: list[str] = []
    effects = cfg.pedalboard.effects
    for index, effect in enumerate(effects):
        if effect.type != "Gain" or not effect.enabled:
            continue
        gain_db = getattr(effect, "gain_db", 0.0)
        if gain_db <= 0:
            continue
        followed = any(
            later.enabled and later.type in _LEVEL_DEPENDENT_TYPES
            for later in effects[index + 1 :]
        )
        if not followed:
            offenders.append(
                f"{label}: effects[{index}] Gain(+{gain_db} dB) has no enabled "
                "level-dependent effect after it"
            )
    return offenders


def _post_placement_offenders(paths: list[Path]) -> set[str]:
    """Config-relative paths that level only after a level-dependent effect.

    A preset is skipped unless it levels at the output, and then flagged when
    any of its resolved conditions leaves an enabled level-dependent effect in
    the chain.
    """
    offenders: set[str] = set()
    for path in paths:
        cfg = load_config(path)
        if cfg.normalisation.placement != "post":
            continue
        has_level_dependent = any(
            effect.enabled and effect.type in _LEVEL_DEPENDENT_TYPES
            for _, resolved in _resolved_conditions(cfg)
            for effect in resolved.pedalboard.effects
        )
        if has_level_dependent:
            try:
                offenders.add(path.relative_to(CONFIG_DIR).as_posix())
            except ValueError:
                offenders.add(str(path))
    return offenders


def test_positive_gain_is_followed_by_a_level_dependent_effect() -> None:
    """No shipped preset boosts the level without a later level-dependent effect."""
    offenders: list[str] = []
    for path in presets():
        cfg = load_config(path)
        label = path.relative_to(CONFIG_DIR).as_posix()
        for name, resolved in _resolved_conditions(cfg):
            offenders.extend(_positive_gain_offenders(resolved, f"{label} [{name}]"))

    assert not offenders, "\n".join(offenders)


def test_positive_gain_rule_flags_a_gain_with_nothing_after_it() -> None:
    """The rule above rejects a positive gain that nothing downstream pins."""
    payload = load_config(_SOURCE_PATH).model_dump(mode="python")
    payload["normalisation"] = {"enabled": True, "placement": "both"}
    payload["pedalboard"]["effects"] = [{"type": "Gain", "gain_db": 6.0, "enabled": True}]
    cfg = PipelineConfig.model_validate(payload)
    assert _positive_gain_offenders(cfg, "synthetic"), (
        "a positive gain with nothing after it must be flagged"
    )

    payload["pedalboard"]["effects"].append(
        {"type": "Distortion", "drive_db": 25.0, "enabled": True}
    )
    cfg = PipelineConfig.model_validate(payload)
    assert not _positive_gain_offenders(cfg, "synthetic"), (
        "a level-dependent effect after the gain must satisfy the rule"
    )


def test_post_placement_with_a_level_dependent_effect_is_allow_listed() -> None:
    """Output-only levelling over a level-dependent effect stays a known to-do."""
    offending = _post_placement_offenders(presets())

    unlisted = sorted(offending - _ALLOWED_POST_PLACEMENT)
    assert unlisted == [], (
        "presets level only after the chain while running a level-dependent "
        "effect and are not on the documented list:\n" + "\n".join(unlisted)
    )
    stale = sorted(_ALLOWED_POST_PLACEMENT - offending)
    assert stale == [], (
        "listed presets no longer level only after a level-dependent effect; "
        "the list must shrink when a file stops offending:\n" + "\n".join(stale)
    )
    assert offending, "the rule matched nothing, so it has no teeth"


def test_paper_presets_share_one_normalisation_section() -> None:
    """Every paper experiment preset carries an identical normalisation section."""
    found = paper_presets()
    assert found, f"no paper experiment presets found under {PAPER_DIR}"

    groups: dict[str, list[str]] = {}
    for path in found:
        key = json.dumps(
            load_config(path).normalisation.model_dump(mode="json"), sort_keys=True
        )
        groups.setdefault(key, []).append(path.relative_to(CONFIG_DIR).as_posix())

    assert len(groups) == 1, "paper presets disagree on normalisation:\n" + "\n".join(
        f"{key}: {names}" for key, names in sorted(groups.items())
    )


def test_source_yaml_documents_the_normalisation_and_clipping_fields() -> None:
    """The reference config comments every normalisation and clipping field."""
    documented = set(
        re.findall(
            r"^[ \t]*#[ \t]+([A-Za-z_][A-Za-z0-9_]*):",
            _SOURCE_PATH.read_text(),
            re.M,
        )
    )
    expected = (
        set(NormalisationSection.model_fields)
        | set(NormalisationStage.model_fields)
        | (set(ClippingConfig.model_fields) - {"type"})
    )
    missing = sorted(expected - documented)
    assert not missing, f"undocumented normalisation/clipping fields: {missing}"


def test_every_gate_and_normalisation_key_is_read_in_src() -> None:
    """No gate or normalisation field is dead config the code never reads."""
    source_text = "\n".join(
        path.read_text() for path in sorted(_SRC_DIR.rglob("*.py"))
    )
    fields = set(QualityGatesSection.model_fields) | set(NormalisationSection.model_fields)
    assert fields, "the section models expose no fields to check"

    unread = sorted(
        field for field in fields if not re.search(rf"\.{field}\b", source_text)
    )
    assert not unread, f"gate/normalisation keys never read in src/: {unread}"
