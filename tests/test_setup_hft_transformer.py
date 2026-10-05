"""Install of the converted hFT-Transformer checkpoint by scripts/setup_hft_transformer.py.

Every test here is offline: the archive is built in a tmp directory and served over a
``file://`` URL or through a fake opener, and the registry entry is monkeypatched to
match whatever was built.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from sonitra.transcribe._hft import checkpoints

REPO = Path(__file__).resolve().parents[1]
REPO_SRC = REPO / "src"
SCRIPT = REPO / "scripts" / "setup_hft_transformer.py"
UPSTREAM_MODEL_SOURCE = (
    REPO / "misc" / "hft_transformer_spike" / "upstream" / "model" / "model_spec2midi.py"
)

MEMBER = "checkpoint/MAESTRO-V3/model_016_003.pkl"
PARAMETER_MEMBER = "checkpoint/MAESTRO-V3/parameter.json"

MANIFEST_KEYS = {
    "checkpoint",
    "upstream_commit",
    "zip_sha256",
    "state_digest",
    "file_sha256",
    "created_utc",
    "sonitra_version",
}


# ── loading the script under test ──────────────────────────────────────


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("setup_hft_transformer_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    if not SCRIPT.is_file():
        pytest.fail(f"{SCRIPT} does not exist")
    return _load_script()


# ── archives served to the script ───────────────────────────────────────


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Archive:
    """A zip the script can be pointed at, with the registry values that match it."""

    path: Path
    sha256: str
    size: int
    state_digest: str

    @property
    def url(self) -> str:
        return self.path.resolve().as_uri()


def _build_zip(path: Path, members: dict[str, bytes]) -> tuple[str, int]:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    data = path.read_bytes()
    return _sha256_bytes(data), len(data)


@pytest.fixture
def dummy_archive(tmp_path: Path) -> Archive:
    """A zip whose bytes are known but whose contents cannot be converted."""
    sha256, size = _build_zip(
        tmp_path / "dummy.zip",
        {MEMBER: b"not a real checkpoint", PARAMETER_MEMBER: b"{}"},
    )
    return Archive(path=tmp_path / "dummy.zip", sha256=sha256, size=size, state_digest="")


#: Builds a tiny upstream-shaped pickle in a child process (the released module source is
#: exec'd under the module name its pickle references) and reports what the registry would
#: pin for it. Needs torch, so only the slow tests use it.
_FABRICATE = '''
import json
import pickle
import sys
import types
from pathlib import Path

import torch

pkl_path, source_path, repo_src = sys.argv[1:4]
sys.path.insert(0, repo_src)
from sonitra.transcribe._hft import checkpoints

source = Path(source_path).read_text()
package = types.ModuleType("model")
package.__path__ = []
module = types.ModuleType("model.model_spec2midi")
module.__file__ = source_path
sys.modules["model"] = package
sys.modules["model.model_spec2midi"] = module
exec(compile(source, source_path, "exec"), module.__dict__)

hparams = {
    "cnn_channel": 2,
    "cnn_kernel": 3,
    "dec_head": 2,
    "dec_layer": 1,
    "dropout": 0.1,
    "enc_head": 2,
    "enc_layer": 1,
    "hid_dim": 16,
    "margin_b": 4,
    "margin_f": 4,
    "n_bins": 8,
    "num_frame": 8,
    "num_note": 4,
    "num_velocity": 8,
    "pf_dim": 32,
}
margin = hparams["margin_b"]
encoder = module.Encoder_SPEC2MIDI(
    n_margin=margin,
    n_frame=hparams["num_frame"],
    n_bin=hparams["n_bins"],
    cnn_channel=hparams["cnn_channel"],
    cnn_kernel=hparams["cnn_kernel"],
    hid_dim=hparams["hid_dim"],
    n_layers=hparams["enc_layer"],
    n_heads=hparams["enc_head"],
    pf_dim=hparams["pf_dim"],
    dropout=hparams["dropout"],
    device="cpu",
)
decoder = module.Decoder_SPEC2MIDI(
    n_frame=hparams["num_frame"],
    n_bin=hparams["n_bins"],
    n_note=hparams["num_note"],
    n_velocity=hparams["num_velocity"],
    hid_dim=hparams["hid_dim"],
    n_layers=hparams["dec_layer"],
    n_heads=hparams["dec_head"],
    pf_dim=hparams["pf_dim"],
    dropout=hparams["dropout"],
    device="cpu",
)
torch.manual_seed(0)
model = module.Model_SPEC2MIDI(encoder, decoder)
model.eval()

with open(pkl_path, "wb") as handle:
    pickle.dump(model, handle, protocol=4)

state = model.state_dict()
print(json.dumps({
    "state_digest": checkpoints.state_digest(state),
    "parameters": sum(int(tensor.numel()) for tensor in state.values()),
    "hparams": hparams,
}))
'''


def _parameter_json(hparams: dict[str, Any], parameters: int) -> str:
    """The released `parameter.json` sections the converter cross-checks."""
    return json.dumps(
        {
            "cnn": {"channel": hparams["cnn_channel"], "kernel": hparams["cnn_kernel"]},
            "transformer": {
                "hid_dim": hparams["hid_dim"],
                "pf_dim": hparams["pf_dim"],
                "encoder": {"n_layer": hparams["enc_layer"], "n_head": hparams["enc_head"]},
                "decoder": {"n_layer": hparams["dec_layer"], "n_head": hparams["dec_head"]},
            },
            "training": {"dropout": hparams["dropout"]},
            "parameters": parameters,
        }
    )


@pytest.fixture(scope="module")
def tiny_archive(tmp_path_factory: pytest.TempPathFactory) -> Archive:
    if not UPSTREAM_MODEL_SOURCE.is_file():
        pytest.skip(f"upstream module source not vendored at {UPSTREAM_MODEL_SOURCE}")
    pytest.importorskip("torch")

    workdir = tmp_path_factory.mktemp("hft_setup_archive")
    pkl_path = workdir / "tiny.pkl"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_SRC), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    result = subprocess.run(
        [sys.executable, "-c", _FABRICATE, str(pkl_path), str(UPSTREAM_MODEL_SOURCE), str(REPO_SRC)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])

    path = workdir / "checkpoint.zip"
    sha256, size = _build_zip(
        path,
        {
            MEMBER: pkl_path.read_bytes(),
            PARAMETER_MEMBER: _parameter_json(report["hparams"], report["parameters"]).encode(),
        },
    )
    return Archive(path=path, sha256=sha256, size=size, state_digest=report["state_digest"])


def _patch_registry(monkeypatch: pytest.MonkeyPatch, archive: Archive, **overrides: Any) -> dict[str, Any]:
    """Point the registry at a locally built archive, optionally corrupting one field."""
    entry: dict[str, Any] = {
        "url": archive.url,
        "zip_sha256": archive.sha256,
        "zip_size": archive.size,
        "member": MEMBER,
        "parameter_member": PARAMETER_MEMBER,
        "state_digest": archive.state_digest,
    }
    entry.update(overrides)
    monkeypatch.setitem(checkpoints.CHECKPOINTS, "maestro", entry)
    return entry


def _install_dir(models_dir: Path) -> Path:
    return models_dir / "hft_transformer" / "maestro"


def _leftovers(models_dir: Path) -> list[str]:
    """Everything left under `hft_transformer/` once a run is over."""
    root = models_dir / "hft_transformer"
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir())


# ── a stand-in for what urlopen returns ─────────────────────────────────


class FakeResponse(io.BytesIO):
    """The readable byte stream the script consumes; a real urlopen result behaves alike."""


def _fake_opener(payload: bytes, calls: list[str]) -> Any:
    def opener(request: Any, *, timeout: float) -> FakeResponse:
        calls.append(getattr(request, "full_url", str(request)))
        return FakeResponse(payload)

    return opener


# ── happy path ──────────────────────────────────────────────────────────


@pytest.mark.slow
def test_installs_converted_checkpoint_and_manifest(
    script: Any, tiny_archive: Archive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "models"

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_OK

    install_dir = _install_dir(models_dir)
    assert (install_dir / "model.pt").is_file()
    manifest = json.loads((install_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["checkpoint"] == "maestro"
    assert manifest["upstream_commit"] == checkpoints.UPSTREAM_COMMIT
    assert manifest["zip_sha256"] == tiny_archive.sha256
    assert manifest["state_digest"] == tiny_archive.state_digest
    assert manifest["file_sha256"] == _sha256_file(install_dir / "model.pt")
    assert datetime.strptime(manifest["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
    assert manifest["sonitra_version"]

    loaded = torch.load(install_dir / "model.pt", weights_only=True, map_location="cpu")
    assert loaded["provenance"]["state_digest"] == tiny_archive.state_digest
    assert set(loaded) == {"format_version", "state_dict", "hparams", "inference", "provenance"}

    # nothing but the lock and the finished install is left behind
    assert _leftovers(models_dir) == [".lock", "maestro"]


@pytest.mark.slow
def test_second_run_is_a_no_op_and_force_rewrites(
    script: Any,
    tiny_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "models"
    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_OK

    install_dir = _install_dir(models_dir)
    watched = [install_dir / "model.pt", install_dir / "manifest.json"]
    before = [(path.stat().st_mtime_ns, path.stat().st_ino) for path in watched]

    capsys.readouterr()
    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_OK
    assert "already installed" in capsys.readouterr().out
    assert [(path.stat().st_mtime_ns, path.stat().st_ino) for path in watched] == before

    capsys.readouterr()
    assert script.main(["--models-dir", str(models_dir), "--force"]) == script.EXIT_OK
    assert "already installed" not in capsys.readouterr().out
    assert [(path.stat().st_mtime_ns, path.stat().st_ino) for path in watched] != before


@pytest.mark.slow
@pytest.mark.parametrize("corrupt", ["unreadable", "stale_file_digest", "stale_state_digest"])
def test_an_invalid_install_is_rebuilt(
    script: Any,
    tiny_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt: str,
) -> None:
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "models"
    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_OK

    install_dir = _install_dir(models_dir)
    manifest_file = install_dir / "manifest.json"
    model_file = install_dir / "model.pt"
    before = (model_file.stat().st_mtime_ns, model_file.stat().st_ino)

    if corrupt == "unreadable":
        manifest_file.write_text("{not json", encoding="utf-8")
    else:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        key = "file_sha256" if corrupt == "stale_file_digest" else "state_digest"
        manifest[key] = "00" * 32
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_OK
    rebuilt = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert rebuilt["state_digest"] == tiny_archive.state_digest
    assert rebuilt["file_sha256"] == _sha256_file(model_file)
    assert (model_file.stat().st_mtime_ns, model_file.stat().st_ino) != before
    assert _leftovers(models_dir) == [".lock", "maestro"]


# ── archive verification failures ──────────────────────────────────────


def test_zip_digest_mismatch_fails_and_installs_nothing(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrong = "ff" * 32
    assert wrong != dummy_archive.sha256
    _patch_registry(monkeypatch, dummy_archive, zip_sha256=wrong)
    models_dir = tmp_path / "models"
    calls: list[str] = []
    monkeypatch.setattr(script, "open_url", _fake_opener(dummy_archive.path.read_bytes(), calls))
    monkeypatch.setattr(script, "RETRY_BACKOFF_SEC", 0.0)

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE
    message = capsys.readouterr().err
    assert wrong in message
    assert dummy_archive.sha256 in message
    assert calls, "the download never happened, so nothing was verified"
    assert not _install_dir(models_dir).exists()
    assert _leftovers(models_dir) == [".lock"]


def test_a_short_download_is_retried_then_reported(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entry = _patch_registry(monkeypatch, dummy_archive)
    models_dir = tmp_path / "models"
    calls: list[str] = []
    truncated = dummy_archive.path.read_bytes()[:32]
    monkeypatch.setattr(script, "open_url", _fake_opener(truncated, calls))
    monkeypatch.setattr(script, "RETRY_BACKOFF_SEC", 0.0)

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE
    message = capsys.readouterr().err
    assert script.DOWNLOAD_ATTEMPTS == 3
    assert len(calls) == script.DOWNLOAD_ATTEMPTS
    assert str(entry["zip_size"]) in message
    assert str(len(truncated)) in message
    assert not _install_dir(models_dir).exists()
    assert _leftovers(models_dir) == [".lock"]


def test_a_registry_pin_the_file_does_not_meet_is_refused(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, dummy_archive, zip_size=7)
    models_dir = tmp_path / "models"
    calls: list[str] = []
    monkeypatch.setattr(script, "open_url", _fake_opener(dummy_archive.path.read_bytes(), calls))
    monkeypatch.setattr(script, "RETRY_BACKOFF_SEC", 0.0)

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE
    assert "7" in capsys.readouterr().err
    assert len(calls) == script.DOWNLOAD_ATTEMPTS
    assert _leftovers(models_dir) == [".lock"]


# ── --archive ───────────────────────────────────────────────────────────


@pytest.mark.slow
def test_the_input_archive_is_never_touched(
    script: Any, tiny_archive: Archive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "models"
    source = tmp_path / "downloaded.zip"
    source.write_bytes(tiny_archive.path.read_bytes())
    before = (source.stat().st_mtime_ns, source.stat().st_size, source.read_bytes())

    argv = ["--models-dir", str(models_dir), "--archive", str(source)]
    assert script.main(argv) == script.EXIT_OK
    assert (source.stat().st_mtime_ns, source.stat().st_size, source.read_bytes()) == before
    assert (_install_dir(models_dir) / "model.pt").is_file()


def test_a_wrong_archive_is_refused_and_left_alone(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, dummy_archive)
    models_dir = tmp_path / "models"
    # one flipped byte keeps the size, so the digest is the only check that can catch it
    corrupted = bytearray(dummy_archive.path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0xFF
    other = tmp_path / "other.zip"
    other.write_bytes(bytes(corrupted))
    other_sha = _sha256_bytes(bytes(corrupted))
    before = other.read_bytes()
    assert other.stat().st_size == dummy_archive.size

    argv = ["--models-dir", str(models_dir), "--archive", str(other)]
    assert script.main(argv) == script.EXIT_FAILURE
    message = capsys.readouterr().err
    assert "sha256 mismatch" in message
    assert dummy_archive.sha256 in message
    assert other_sha in message
    assert other.read_bytes() == before
    assert not _install_dir(models_dir).exists()
    assert _leftovers(models_dir) == [".lock"]


def test_an_archive_that_is_not_a_zip_is_refused(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, dummy_archive)
    models_dir = tmp_path / "models"
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"definitely not a zip file" * 8)
    _patch_registry(
        monkeypatch,
        dummy_archive,
        url=junk.resolve().as_uri(),
        zip_sha256=_sha256_bytes(junk.read_bytes()),
        zip_size=junk.stat().st_size,
    )

    assert script.main(["--models-dir", str(models_dir), "--archive", str(junk)]) == script.EXIT_FAILURE
    assert "zip" in capsys.readouterr().err
    assert _leftovers(models_dir) == [".lock"]


# ── torch must be present before anything is fetched ────────────────────


def test_a_missing_torch_exits_three_without_touching_the_network(
    script: Any,
    tiny_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "models"
    calls: list[str] = []
    monkeypatch.setattr(script, "open_url", _fake_opener(tiny_archive.path.read_bytes(), calls))

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        script.importlib.util,
        "find_spec",
        lambda name, package=None: None if name == "torch" else real_find_spec(name, package),
    )

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_NO_TORCH
    assert script.EXIT_NO_TORCH == 3
    message = capsys.readouterr().err
    assert "torch" in message
    assert "transkun" in message
    assert calls == [], "the download ran before the torch check"
    assert not (models_dir / "hft_transformer").exists()


# ── where the weights go ────────────────────────────────────────────────


def test_models_dir_precedence(script: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("SONITRA_MODELS_DIR", raising=False)

    assert script.resolve_models_dir(None) == home / ".cache" / "sonitra" / "models"
    assert script.resolve_models_dir(None) == checkpoints.models_dir()

    monkeypatch.setenv("SONITRA_MODELS_DIR", str(tmp_path / "from_env"))
    assert script.resolve_models_dir(None) == tmp_path / "from_env"
    assert script.resolve_models_dir(None) == checkpoints.models_dir()

    assert script.resolve_models_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert script.resolve_models_dir(None) == tmp_path / "from_env"


@pytest.mark.slow
def test_the_environment_is_used_when_no_flag_is_given(
    script: Any, tiny_archive: Archive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "env_models"
    monkeypatch.setenv("SONITRA_MODELS_DIR", str(models_dir))

    assert script.main([]) == script.EXIT_OK
    assert (_install_dir(models_dir) / "model.pt").is_file()


# ── locking and atomicity ──────────────────────────────────────────────


def test_the_install_lock_is_created_and_a_second_holder_waits(
    script: Any, dummy_archive: Archive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_registry(monkeypatch, dummy_archive)
    models_dir = tmp_path / "models"
    monkeypatch.setattr(script, "open_url", _fake_opener(dummy_archive.path.read_bytes(), []))
    # the dummy contents cannot be converted, so the run fails after taking the lock
    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE

    lock_file = models_dir / "hft_transformer" / ".lock"
    assert lock_file.is_file()

    competitor = lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(script.SetupError):
            script.acquire_lock(lock_file, timeout=0.2)
    finally:
        competitor.close()

    handle = script.acquire_lock(lock_file, timeout=0.2)
    try:
        with pytest.raises(script.SetupError):
            script.acquire_lock(lock_file, timeout=0.2)
    finally:
        handle.close()


def test_a_failure_mid_install_leaves_no_weights_behind(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, dummy_archive)
    models_dir = tmp_path / "models"
    monkeypatch.setattr(script, "open_url", _fake_opener(dummy_archive.path.read_bytes(), []))

    def fake_convert(pickle_path: Any, out_path: Any, **kwargs: Any) -> dict[str, Any]:
        """The weights are written; the manifest write underneath them then fails."""
        Path(out_path).write_bytes(b"converted weights")
        return {"provenance": {"state_digest": dummy_archive.state_digest}}

    def boom(*args: Any, **kwargs: Any) -> None:
        raise script.SetupError("injected manifest failure")

    monkeypatch.setattr(script, "convert_checkpoint", fake_convert)
    monkeypatch.setattr(script, "write_manifest", boom)

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE
    assert "injected manifest failure" in capsys.readouterr().err
    assert not _install_dir(models_dir).exists(), "a half-installed directory survived"
    assert not list(models_dir.rglob("model.pt")), "weights survived a failed install"
    assert _leftovers(models_dir) == [".lock"], _leftovers(models_dir)


@pytest.mark.slow
def test_a_successful_install_is_published_with_one_rename(
    script: Any, tiny_archive: Archive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_registry(monkeypatch, tiny_archive)
    models_dir = tmp_path / "models"
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_OK

    install_dir = _install_dir(models_dir)
    published = [(src, dst) for src, dst in replaced if dst == str(install_dir)]
    assert len(published) == 1, replaced
    staged = Path(published[0][0])
    assert staged.parent == models_dir / "hft_transformer"
    assert staged.name.startswith(".maestro"), staged.name
    assert staged != install_dir
    assert not staged.exists(), "the staging directory outlived the rename"


# ── conversion failures ─────────────────────────────────────────────────


@pytest.mark.slow
def test_a_conversion_failure_installs_nothing(
    script: Any,
    tiny_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, tiny_archive, state_digest="00" * 32)
    models_dir = tmp_path / "models"

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE
    message = capsys.readouterr().err
    assert "state digest mismatch" in message
    assert "00" * 32 in message
    assert not _install_dir(models_dir).exists()
    assert _leftovers(models_dir) == [".lock"]


def test_malformed_weights_are_reported_as_a_setup_failure(
    script: Any,
    dummy_archive: Archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_registry(monkeypatch, dummy_archive, state_digest="00" * 32)
    models_dir = tmp_path / "models"
    monkeypatch.setattr(script, "open_url", _fake_opener(dummy_archive.path.read_bytes(), []))

    assert script.main(["--models-dir", str(models_dir)]) == script.EXIT_FAILURE
    assert capsys.readouterr().err.strip(), "a malformed pickle produced no message"
    assert _leftovers(models_dir) == [".lock"]


# ── the CLI surface ─────────────────────────────────────────────────────


def test_unknown_checkpoint_names_the_valid_ones(
    script: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert script.main(["--checkpoint", "nope"]) == script.EXIT_FAILURE
    message = capsys.readouterr().err
    assert "nope" in message
    for name in checkpoints.CHECKPOINTS:
        assert name in message


def test_help_exits_zero_and_prints_the_examples() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT.relative_to(REPO)), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "python scripts/setup_hft_transformer.py" in result.stdout


def test_a_bad_flag_is_an_argparse_usage_error() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT.relative_to(REPO)), "--nope"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 2
