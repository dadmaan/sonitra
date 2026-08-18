from __future__ import annotations

from sonitra.benchmark.host_info import collect_host_info


def test_collect_host_info_returns_documented_keys() -> None:
    info = collect_host_info()
    for key in (
        "cpu_model",
        "cpu_count",
        "ram_bytes",
        "gpu",
        "os",
        "python",
        "packages",
    ):
        assert key in info, f"missing key {key!r}"
    assert isinstance(info["cpu_model"], str) and info["cpu_model"]
    assert isinstance(info["cpu_count"], int) and info["cpu_count"] >= 0
    assert info["ram_bytes"] is None or isinstance(info["ram_bytes"], int)
    assert isinstance(info["gpu"], list)
    assert isinstance(info["os"], str) and info["os"]
    assert isinstance(info["python"], str) and info["python"]


def test_collect_host_info_packages_is_dict() -> None:
    packages = collect_host_info()["packages"]
    assert isinstance(packages, dict)
    for name, version in packages.items():
        assert isinstance(name, str)
        assert isinstance(version, str) and version


def test_collect_host_info_never_raises() -> None:
    # Called twice to be safe; it must never raise on either call.
    assert collect_host_info() is not None
    assert collect_host_info() is not None
