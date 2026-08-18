from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from typing import Any


def collect_host_info() -> dict[str, Any]:
    """Probe the host environment for benchmark reporting.

    Every probe is individually wrapped in try/except so this function never
    raises: missing packages, unsupported platforms, and failed subprocesses
    all degrade to fallback values.

    Returns a dict with keys ``cpu_model``, ``cpu_count``, ``ram_bytes``,
    ``gpu``, ``os``, ``python`` and ``packages``.
    """
    # CPU model: first "model name" line from /proc/cpuinfo (Linux); fall back
    # to platform.processor(), then platform.machine().
    cpu_model = ""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:  # noqa: BLE001 - never raise
            cpu_model = ""
    if not cpu_model:
        cpu_model = platform.processor()
    if not cpu_model:
        cpu_model = platform.machine()

    # CPU count: os.cpu_count(); None -> 1, hard failure -> 0.
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:  # noqa: BLE001 - never raise
        cpu_count = 0

    # RAM bytes: sysconf-based, Linux-only.
    ram_bytes: int | None = None
    if platform.system() == "Linux":
        try:
            ram_bytes = int(
                os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            )
        except (AttributeError, OSError, ValueError):
            ram_bytes = None

    # GPU: nvidia-smi first (cheap), then tensorflow, then torch. The heavy
    # package imports only run when earlier probes produced nothing.
    gpu: list[str] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=5,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpu = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
    except Exception:  # noqa: BLE001 - never raise
        gpu = []
    if not gpu:
        try:
            import tensorflow as tf  # type: ignore[import-not-found]

            gpu = [device.name for device in tf.config.list_physical_devices("GPU")]
        except Exception:  # noqa: BLE001 - never raise
            gpu = []
    if not gpu:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                gpu = [
                    torch.cuda.get_device_name(device_index)
                    for device_index in range(torch.cuda.device_count())
                ]
        except Exception:  # noqa: BLE001 - never raise
            gpu = []

    # Installed package versions (only those that are importable/installed).
    packages: dict[str, str] = {}
    for name in ("sonitra", "numpy", "scipy", "basic-pitch", "tensorflow", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:  # noqa: BLE001 - never raise
            continue

    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "ram_bytes": ram_bytes,
        "gpu": gpu,
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
    }
