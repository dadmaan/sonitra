#!/usr/bin/env python3
r"""Download and convert the pinned hFT-Transformer checkpoint.

Installs the released weights once as a `weights_only=True`-loadable state dict and
records the provenance next to them. The archive's size and sha256 are checked against
the registry before anything is converted, and an exclusive lock keeps two runs from
interleaving. Exits 1 on failure, 2 on a usage error and 3 when torch is missing.

Examples:
    # default install into ~/.cache/sonitra/models
    python scripts/setup_hft_transformer.py

    # install into a container-visible directory
    python scripts/setup_hft_transformer.py --models-dir /models

    # install from an archive you already downloaded (offline)
    python scripts/setup_hft_transformer.py --archive /tmp/checkpoint.zip
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from sonitra.transcribe._hft import checkpoints
from sonitra.transcribe._hft.convert import ConversionError, convert_checkpoint

#: Container entrypoints branch on these: 0 done, 1 broken, 2 misuse, 3 no torch yet.
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_NO_TORCH = 3

#: The release is a single GitHub asset behind a redirect, so a retry covers a truncated
#: or corrupted transfer; three attempts is what the container start-up budget allows.
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SEC = 60.0
RETRY_BACKOFF_SEC = 0.5
CHUNK_BYTES = 1 << 20
LOCK_POLL_SEC = 0.05

LOCK_NAME = ".lock"
MODEL_NAME = "model.pt"
MANIFEST_NAME = "manifest.json"

#: Some proxies reject a download without one, and the release sends no digest header.
USER_AGENT = "sonitra-setup-hft-transformer/1 (+python scripts/setup_hft_transformer.py)"


class SetupError(Exception):
    """The install could not be completed; the message names what to do about it."""


class MissingTorchError(SetupError):
    """torch is absent, so there is nothing this script could convert the weights with."""


class FetchError(SetupError):
    """The pinned archive could not be obtained, or does not match its pinned digest."""


def log(message: str) -> None:
    """One line of progress on stdout; the container logs capture both streams."""
    print(message, flush=True)


def warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sonitra_version() -> str:
    """Installed version of the package, or ``unknown`` from a source checkout."""
    try:
        from importlib.metadata import version

        return version("sonitra")
    except Exception:
        return "unknown"


# ── output helpers ──────────────────────────────────────────────────────


class _Progress:
    """Byte progress for one download: a rich bar when rich is importable, else a line.

    rich is optional, so a missing or broken install must not stop the download.
    """

    def __init__(self, description: str, total: int) -> None:
        self._description = description
        self._total = total
        self._bar: Any = None
        self._task = 0

    def __enter__(self) -> _Progress:
        try:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )
        except Exception:
            log(f"{self._description} {self._total} bytes")
            return self
        self._bar = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        self._task = self._bar.add_task(self._description, total=self._total or None)
        self._bar.start()
        return self

    def update(self, completed: int) -> None:
        if self._bar is not None:
            self._bar.update(self._task, completed=completed)

    def __exit__(self, *exc_info: object) -> None:
        if self._bar is not None:
            self._bar.stop()


# ── where the weights go ────────────────────────────────────────────────


def resolve_models_dir(explicit: Path | str | None = None) -> Path:
    """Install root: the flag wins, then ``SONITRA_MODELS_DIR``, then the user cache."""
    if explicit is not None:
        return Path(explicit).expanduser()
    return checkpoints.models_dir()


def install_root(models_dir: Path) -> Path:
    """Directory holding every hft_transformer install, and the install lock."""
    return Path(models_dir) / "hft_transformer"


def registry_entry(checkpoint: str) -> dict[str, Any]:
    """The pinned release metadata for ``checkpoint``; a SetupError names the valid names."""
    try:
        return checkpoints.checkpoint_entry(checkpoint)
    except KeyError as exc:
        raise SetupError(str(exc.args[0])) from exc


def require_torch() -> None:
    """Refuse to start without torch, before a single byte is fetched.

    An image that was built without the extra must not depend on the network to learn
    that, so the check is ``find_spec`` rather than an import.
    """
    if importlib.util.find_spec("torch") is not None:
        return
    raise MissingTorchError(
        "torch is not installed, and converting the released checkpoint needs it. "
        "Install the extra that provides it first: `uv sync --extra transkun --extra dev` "
        '(or `pip install -e ".[transkun,dev]"`), then run python scripts/setup_hft_transformer.py again.'
    )


# ── locking ─────────────────────────────────────────────────────────────


def acquire_lock(lock_file: Path, *, timeout: float | None = None) -> IO[str]:
    """Take the exclusive install lock and hold it until the returned handle is closed.

    A devcontainer rebuild and a production container can share one models directory, so
    the lock is what keeps two conversions from writing the same directory at once. With
    ``timeout`` set, waiting is bounded and a lock that stays held raises ``SetupError``.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_file.open("a+", encoding="utf-8")
    deadline = None if timeout is None else time.monotonic() + timeout
    announced = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            if deadline is not None and time.monotonic() >= deadline:
                handle.close()
                raise SetupError(
                    f"another run still holds {lock_file}; wait for it to finish, then try again"
                ) from None
            if not announced:
                announced = True
                log(f"another run holds {lock_file}; waiting for it to finish")
            time.sleep(LOCK_POLL_SEC)


# ── is there already a usable install? ──────────────────────────────────


def read_manifest(manifest_file: Path) -> dict[str, Any]:
    """Parse an install manifest, refusing anything that is not one."""
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SetupError(f"{manifest_file} is not readable JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SetupError(f"{manifest_file} does not hold a manifest object")
    return manifest


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, read in chunks so a 20 MB file costs nothing extra."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(
    install_dir: Path,
    entry: Mapping[str, Any],
    *,
    checkpoint: str = checkpoints.DEFAULT_CHECKPOINT,
    force: bool = False,
) -> tuple[bool, str]:
    """``(intact, state digest)`` when the install can be kept, else ``(False, reason)``.

    The recorded tensor digest is compared with the pinned one rather than recomputed:
    reloading the weights on every run would cost a full read of the state dict, and the
    file digest below already proves the bytes are the ones that were converted.
    """
    if force:
        return False, "--force was given, so the existing install is rebuilt"
    manifest_file = install_dir / MANIFEST_NAME
    model_file = install_dir / MODEL_NAME
    if not manifest_file.is_file():
        return False, f"{install_dir} has no {MANIFEST_NAME}"
    try:
        manifest = read_manifest(manifest_file)
    except SetupError as exc:
        return False, str(exc)
    if not model_file.is_file():
        return False, f"{install_dir} has no {MODEL_NAME}"
    if manifest.get("checkpoint") != checkpoint:
        return False, f"{manifest_file} records checkpoint {manifest.get('checkpoint')!r}"
    if manifest.get("state_digest") != entry["state_digest"]:
        return False, f"{manifest_file} does not record the pinned state digest"
    if manifest.get("zip_sha256") != entry["zip_sha256"]:
        return False, f"{manifest_file} does not record the pinned archive digest"
    actual = file_sha256(model_file)
    if manifest.get("file_sha256") != actual:
        return False, f"{model_file} does not match the digest recorded in {manifest_file}"
    return True, str(manifest["state_digest"])


# ── obtaining the archive ───────────────────────────────────────────────


def open_url(request: urllib.request.Request, *, timeout: float) -> Any:
    """Open one URL. urllib follows the GitHub release redirect by itself."""
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 (https release asset)


def _archive_problem(size: int, sha256: str, entry: Mapping[str, Any]) -> str | None:
    """Why ``size``/``sha256`` disagree with the pinned release, or None when they agree."""
    expected_size = int(entry["zip_size"])
    if size != expected_size:
        return f"size mismatch: expected {expected_size} bytes, got {size}"
    expected_sha = str(entry["zip_sha256"])
    if sha256 != expected_sha:
        return f"sha256 mismatch: expected {expected_sha}, got {sha256}"
    return None


def verify_archive(path: Path, entry: Mapping[str, Any]) -> None:
    """Check an already downloaded archive; the file is only ever read."""
    if not path.is_file():
        raise FetchError(f"{path} does not exist")
    problem = _archive_problem(path.stat().st_size, file_sha256(path), entry)
    if problem is not None:
        raise FetchError(f"{path} is not the pinned release archive: {problem}")


def _stream_download(
    destination: Path, entry: Mapping[str, Any], opener: Callable[..., Any], timeout: float
) -> None:
    """Write the release asset to ``destination``, hashing it as it lands."""
    request = urllib.request.Request(str(entry["url"]), headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    written = 0
    with opener(request, timeout=timeout) as response, destination.open("wb") as handle:
        with _Progress(f"downloading {entry['url']}", int(entry["zip_size"])) as progress:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                progress.update(written)
    problem = _archive_problem(written, digest.hexdigest(), entry)
    if problem is not None:
        destination.unlink(missing_ok=True)
        raise FetchError(f"{entry['url']} is not the pinned release archive: {problem}")


def fetch_archive(
    destination: Path,
    entry: Mapping[str, Any],
    *,
    opener: Callable[..., Any] | None = None,
    attempts: int = DOWNLOAD_ATTEMPTS,
    timeout: float = DOWNLOAD_TIMEOUT_SEC,
    backoff: float = RETRY_BACKOFF_SEC,
) -> None:
    """Download the pinned archive to ``destination`` and verify it, or raise ``FetchError``.

    A size or digest mismatch is retried like a network error: a partial or corrupted
    transfer is indistinguishable from a bad mirror, and this runs once per install.
    """
    open_it = opener or open_url
    destination.parent.mkdir(parents=True, exist_ok=True)
    problem = f"{entry['url']} was never fetched"
    for attempt in range(1, attempts + 1):
        try:
            _stream_download(destination, entry, open_it, timeout)
            return
        except (FetchError, urllib.error.URLError, OSError, TimeoutError) as exc:
            problem = str(exc)
        destination.unlink(missing_ok=True)
        if attempt < attempts:
            log(f"attempt {attempt}/{attempts} failed: {problem}; retrying")
            time.sleep(backoff * attempt)
    raise FetchError(f"gave up on {entry['url']} after {attempts} attempts: {problem}")


def extract_members(
    archive: Path, entry: Mapping[str, Any], destination: Path
) -> tuple[Path, Path]:
    """Copy the two registered members out of the archive as ``(pickle, parameter.json)``.

    Each member is written under a name of our own rather than its path in the archive,
    so a member name can never write outside the destination directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for key, filename in (("member", "checkpoint.pkl"), ("parameter_member", "parameter.json")):
                member = str(entry[key])
                if member not in names:
                    raise SetupError(
                        f"{archive} has no {member}, which the registry pins for this release"
                    )
                target = destination / filename
                with zf.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle, CHUNK_BYTES)
                extracted[key] = target
    except (zipfile.BadZipFile, OSError) as exc:
        raise SetupError(f"{archive} is not a readable zip archive: {exc}") from exc
    return extracted["member"], extracted["parameter_member"]


# ── writing the install ─────────────────────────────────────────────────


def build_manifest(
    checkpoint: str,
    entry: Mapping[str, Any],
    *,
    state_digest: str,
    file_digest: str,
) -> dict[str, Any]:
    """The provenance recorded next to the weights; its digests gate the next run."""
    return {
        "checkpoint": checkpoint,
        "upstream_commit": checkpoints.UPSTREAM_COMMIT,
        "zip_sha256": str(entry["zip_sha256"]),
        "state_digest": state_digest,
        "file_sha256": file_digest,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sonitra_version": sonitra_version(),
    }


def write_manifest(manifest_file: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest with a trailing newline so the file stays diff-friendly."""
    text = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    manifest_file.write_text(text, encoding="utf-8")


def publish(staged: Path, install_dir: Path) -> None:
    """Move the finished staging directory into place with a single rename."""
    # mkdtemp is private by default, and a models directory is shared by every user that
    # reads the weights, so the published directory gets the usual mode.
    staged.chmod(0o755)
    # os.replace cannot overwrite a non-empty directory, so an install that failed
    # validation is removed first. The gap exposes no directory at all rather than a
    # half-written one, which is what the backend refuses to load.
    shutil.rmtree(install_dir, ignore_errors=True)
    os.replace(staged, install_dir)


def install(
    models_dir: Path,
    checkpoint: str,
    entry: Mapping[str, Any],
    *,
    archive: Path | None = None,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fetch, verify, convert and publish one checkpoint; return the manifest written.

    The download and the staging directory both live under the models directory so the
    final rename is a rename within one filesystem, and every temporary directory is
    removed again whether or not the install succeeds.
    """
    root = install_root(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    install_dir = checkpoints.checkpoint_dir(checkpoint, models_dir)
    work = Path(tempfile.mkdtemp(dir=str(root), prefix=f".{checkpoint}-work-"))
    staged = Path(tempfile.mkdtemp(dir=str(root), prefix=f".{checkpoint}-stage-"))
    try:
        if archive is None:
            source = work / "checkpoint.zip"
            fetch_archive(source, entry, opener=opener)
        else:
            source = Path(archive)
            verify_archive(source, entry)
        pickle_path, parameter_path = extract_members(source, entry, work / "extracted")
        model_file = staged / MODEL_NAME
        try:
            payload = convert_checkpoint(
                pickle_path,
                model_file,
                checkpoint=checkpoint,
                parameter_path=parameter_path,
                expected_state_digest=str(entry["state_digest"]),
            )
        except ConversionError as exc:
            raise SetupError(f"converting the {checkpoint} checkpoint failed: {exc}") from exc
        except Exception as exc:
            # A malformed pickle or a torch failure is a setup failure too, and the
            # converter already refuses anything it cannot vouch for.
            raise SetupError(
                f"converting the {checkpoint} checkpoint failed: {type(exc).__name__}: {exc}"
            ) from exc
        manifest = build_manifest(
            checkpoint,
            entry,
            state_digest=str(payload["provenance"]["state_digest"]),
            file_digest=file_sha256(model_file),
        )
        write_manifest(staged / MANIFEST_NAME, manifest)
        publish(staged, install_dir)
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(staged, ignore_errors=True)


# ── the command line ────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Parser whose help carries the module docstring, examples included."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="directory to install into (default: SONITRA_MODELS_DIR or ~/.cache/sonitra/models)",
    )
    parser.add_argument(
        "--checkpoint",
        default=checkpoints.DEFAULT_CHECKPOINT,
        help="registered checkpoint name (default: %(default)s)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="use an already downloaded release archive instead of fetching it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="convert and install again even when a valid install is present",
    )
    return parser


def _fail(error: Exception) -> int:
    warn(f"error: {error}")
    return EXIT_FAILURE


def main(argv: Sequence[str] | None = None) -> int:
    """Install the checkpoint; the return value is the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        entry = registry_entry(args.checkpoint)
    except SetupError as exc:
        return _fail(exc)
    try:
        require_torch()
    except MissingTorchError as exc:
        warn(f"error: {exc}")
        return EXIT_NO_TORCH

    models_dir = resolve_models_dir(args.models_dir)
    install_dir = checkpoints.checkpoint_dir(args.checkpoint, models_dir)
    try:
        lock = acquire_lock(install_root(models_dir) / LOCK_NAME)
    except SetupError as exc:
        return _fail(exc)
    try:
        intact, detail = is_installed(
            install_dir, entry, checkpoint=args.checkpoint, force=args.force
        )
        if intact:
            log(f"{args.checkpoint} already installed at {install_dir} (state digest {detail}).")
            log("Rebuild it with: python scripts/setup_hft_transformer.py --force")
            return EXIT_OK
        source = str(args.archive) if args.archive is not None else str(entry["url"])
        log(f"installing {args.checkpoint} from {source} into {install_dir} ({detail})")
        manifest = install(models_dir, args.checkpoint, entry, archive=args.archive)
    except SetupError as exc:
        return _fail(exc)
    finally:
        lock.close()

    log(f"installed {args.checkpoint} at {install_dir}")
    log(f"  state digest {manifest['state_digest']}")
    log(f"  weights {install_dir / MODEL_NAME}, provenance {install_dir / MANIFEST_NAME}")
    log("the hft_transformer transcriber loads these weights; nothing else has to be set.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
