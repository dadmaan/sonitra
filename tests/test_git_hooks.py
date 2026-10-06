"""Decisions the pre-push hook makes: which pushes it checks, and how it refuses.

Every case builds a throwaway repository and a fake ``docker`` that records its
argv and environment and answers from files the test writes, so no real
container, no real git config and no real push is involved.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="git and bash are needed to exercise the pre-push hook",
)

_HOOK_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-push"
_ORIGIN_URL = "https://example.invalid/sonitra.git"
_ZERO_SHA = "0" * 40
_BYPASS = "bypass in an emergency: git push --no-verify"
#: Ends one invocation in the fake docker's log; docker args never span lines.
_CALL_MARKER = "-- end of call --"

#: Bash builtins only, so the fake needs nothing from PATH: it records its argv
#: in ``$FAKE_DOCKER_LOG``, its MSYS_NO_PATHCONV in ``$FAKE_DOCKER_ENVLOG``, and
#: reads its answers from ``$FAKE_DOCKER_CONFIG``.
_FAKE_DOCKER = """#!/usr/bin/env bash
# Fake docker: record argv and environment, then answer from the files the test writes.
set -euo pipefail
config="${FAKE_DOCKER_CONFIG:?}"
printf '%s\\n' "$@" >>"$FAKE_DOCKER_LOG"
printf '%s\\n' '@MARKER@' >>"$FAKE_DOCKER_LOG"
printf '%s=%s\\n' "${1:-}" "${MSYS_NO_PATHCONV-<unset>}" >>"$FAKE_DOCKER_ENVLOG"
configured_code() {
  local code=0
  if [ -f "$config/$1" ]; then read -r code <"$config/$1" || true; fi
  printf '%s' "$code"
}
case "${1:-}" in
ps)
  code="$(configured_code ps_exit)"
  if [ "$code" -ne 0 ]; then
    echo 'Cannot connect to the Docker daemon.' >&2
    exit "$code"
  fi
  if [ -f "$config/ps_names" ]; then
    while IFS= read -r name; do printf '%s\\n' "$name"; done <"$config/ps_names"
  fi
  ;;
exec)
  exit "$(configured_code exec_exit)"
  ;;
*)
  echo "fake docker: unexpected subcommand '${1:-}'" >&2
  exit 64
  ;;
esac
"""

#: What the "no docker on PATH" case leaves reachable: a git and a bash for the
#: hook itself, plus the coreutils a shell may reach for.
_PATH_TOOLS = (
    "bash",
    "cat",
    "cut",
    "env",
    "git",
    "grep",
    "head",
    "ls",
    "sed",
    "sort",
    "tail",
    "tr",
    "wc",
)


class Hook:
    """A throwaway repo holding the hook, plus a fake ``docker`` on its PATH."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.bin = root / "bin"
        self.docker_log = root / "docker.log"
        self.docker_env_log = root / "docker-env.log"
        self.docker_config = root / "docker-config"
        self.hook = self.repo / "scripts" / "hooks" / "pre-push"
        self.home = root / "home"

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, root: Path) -> Hook:
        """``main`` sits on the first commit, ``dev`` (checked out) one commit ahead."""
        case = cls(root)
        for directory in (case.bin, case.docker_config, case.home):
            directory.mkdir(parents=True)
        fake = case.bin / "docker"
        fake.write_text(
            _FAKE_DOCKER.replace("@MARKER@", _CALL_MARKER), encoding="utf-8"
        )
        fake.chmod(0o755)

        case.repo.mkdir()
        case.git("init", "-q")
        case.git("checkout", "-q", "-b", "main")
        (case.repo / "README.md").write_text("sonitra\n", encoding="utf-8")
        # The hook is tracked on main too: the tests check out main as well as
        # dev, and a checkout must not delete the hook under test.
        case.hook.parent.mkdir(parents=True)
        shutil.copyfile(_HOOK_SOURCE, case.hook)
        case.hook.chmod(0o755)
        case.git("add", "-A")
        case.git("commit", "-q", "-m", "first")
        case.git("branch", "dev")
        case.git("checkout", "-q", "dev")
        (case.repo / "dev.txt").write_text("work in progress\n", encoding="utf-8")
        case.git("add", "-A")
        case.git("commit", "-q", "-m", "second")
        return case

    # -- repository ------------------------------------------------------
    def git(self, *args: str) -> str:
        """Run git in the throwaway repo and return its stdout."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=self.env(),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()

    def sha(self, branch: str) -> str:
        return self.git("rev-parse", branch)

    def checkout(self, branch: str) -> None:
        self.git("checkout", "-q", branch)

    def make_dirty(self) -> None:
        (self.repo / "dev.txt").write_text("uncommitted\n", encoding="utf-8")

    def set_container_config(self, name: str) -> None:
        self.git("config", "sonitra.checkContainer", name)

    # -- fake docker -----------------------------------------------------
    def set_container_names(self, *names: str) -> None:
        (self.docker_config / "ps_names").write_text(
            "".join(f"{name}\n" for name in names), encoding="utf-8"
        )

    def set_ps_exit(self, code: int) -> None:
        (self.docker_config / "ps_exit").write_text(f"{code}\n", encoding="utf-8")

    def set_exec_exit(self, code: int) -> None:
        (self.docker_config / "exec_exit").write_text(f"{code}\n", encoding="utf-8")

    # -- invocation ------------------------------------------------------
    def env(self, *, docker_on_path: bool = True) -> dict[str, str]:
        """A child environment that cannot read the developer's own git config."""
        env = {
            key: value
            for key, value in os.environ.items()
            # MSYS_NO_PATHCONV has to start unset, or a hook that never sets it
            # would look correct to the test that requires it.
            if not key.startswith(("GIT_", "DOCKER_", "MSYS_"))
        }
        if docker_on_path:
            path = [str(self.bin), os.environ.get("PATH", "")]
        else:
            self._link_tools()
            path = [str(self._bin_without_docker)]
        env.update(
            {
                "PATH": os.pathsep.join(path),
                "HOME": str(self.home),
                "USERPROFILE": str(self.home),
                "XDG_CONFIG_HOME": str(self.home / "xdg"),
                "GIT_CONFIG_GLOBAL": str(self.home / "gitconfig.global"),
                "GIT_CONFIG_SYSTEM": str(self.home / "gitconfig.system"),
                "GIT_TERMINAL_PROMPT": "0",
                # A shell that cannot honour LC_ALL warns on every start, and
                # those warnings would show up as container names.
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_AUTHOR_NAME": "Hook Test",
                "GIT_AUTHOR_EMAIL": "hook@example.invalid",
                "GIT_COMMITTER_NAME": "Hook Test",
                "GIT_COMMITTER_EMAIL": "hook@example.invalid",
                "FAKE_DOCKER_LOG": str(self.docker_log),
                "FAKE_DOCKER_ENVLOG": str(self.docker_env_log),
                "FAKE_DOCKER_CONFIG": str(self.docker_config),
            }
        )
        return env

    def _link_tools(self) -> None:
        bin_dir = self._bin_without_docker
        bin_dir.mkdir(exist_ok=True)
        for name in _PATH_TOOLS:
            found = shutil.which(name)
            if found is not None:
                (bin_dir / name).symlink_to(found)

    @property
    def _bin_without_docker(self) -> Path:
        return self.root / "bin-without-docker"

    def run(
        self, stdin: str, *, docker_on_path: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run the hook the way git does: argv plus one ref per line on stdin."""
        return subprocess.run(
            [
                shutil.which("bash") or "bash",
                str(self.hook),
                "origin",
                _ORIGIN_URL,
            ],
            cwd=self.repo,
            input=stdin,
            env=self.env(docker_on_path=docker_on_path),
            capture_output=True,
            text=True,
            timeout=120,
        )

    def push(
        self, *lines: str, docker_on_path: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Push ``lines`` (``<local_ref> <sha> <remote_ref> <sha>`` each)."""
        return self.run(
            "".join(f"{line}\n" for line in lines), docker_on_path=docker_on_path
        )

    def branch_line(self, branch: str, local_sha: str | None = None) -> str:
        sha = self.sha(branch) if local_sha is None else local_sha
        return f"refs/heads/{branch} {sha} refs/heads/{branch} {_ZERO_SHA}"

    # -- recorded docker calls -------------------------------------------
    def calls(self, subcommand: str | None = None) -> list[list[str]]:
        """Every recorded ``docker`` invocation, one argv per record."""
        if not self.docker_log.exists():
            return []
        records: list[list[str]] = []
        current: list[str] = []
        for line in self.docker_log.read_text(encoding="utf-8").splitlines():
            if line == _CALL_MARKER:
                records.append(current)
                current = []
            else:
                current.append(line)
        if current:
            records.append(current)
        if subcommand is not None:
            records = [call for call in records if call[:1] == [subcommand]]
        return records

    def exec_args(self) -> list[str]:
        calls = self.calls("exec")
        assert len(calls) == 1, f"expected exactly one docker exec, got {calls}"
        return calls[0]

    def env_log(self) -> list[str]:
        """``<subcommand>=<MSYS_NO_PATHCONV>`` for every recorded call."""
        if not self.docker_env_log.exists():
            return []
        return self.docker_env_log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def hook(tmp_path: Path) -> Hook:
    return Hook.build(tmp_path)


def flag_value(args: list[str], flag: str) -> str | None:
    """The argument following ``flag``, or ``None`` when it is absent."""
    return args[args.index(flag) + 1] if flag in args else None


def output(result: subprocess.CompletedProcess[str]) -> str:
    """Whatever the hook said, whichever stream it chose."""
    return result.stdout + result.stderr


def assert_refused(result: subprocess.CompletedProcess[str]) -> str:
    text = output(result)
    assert result.returncode != 0, f"the push was allowed: {text}"
    assert _BYPASS in text, f"no emergency bypass offered: {text}"
    return text


# -- pushes that are not checked ------------------------------------------------


def test_tag_only_push_is_not_checked(hook: Hook) -> None:
    result = hook.push(
        f"refs/tags/v0.4.0 {hook.sha('dev')} refs/tags/v0.4.0 {_ZERO_SHA}"
    )
    assert result.returncode == 0, output(result)
    assert hook.calls() == []


def test_feature_branch_push_is_not_checked(hook: Hook) -> None:
    result = hook.push(hook.branch_line("feature/x", hook.sha("dev")))
    assert result.returncode == 0, output(result)
    assert hook.calls() == []


def test_deleting_dev_is_not_checked(hook: Hook) -> None:
    result = hook.push(f"refs/heads/dev {_ZERO_SHA} refs/heads/dev {hook.sha('dev')}")
    assert result.returncode == 0, output(result)
    assert hook.calls() == []


# -- pushes that are checked ----------------------------------------------------


def test_dev_at_head_runs_the_fast_checks_in_the_container(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    result = hook.push(hook.branch_line("dev"))
    text = output(result)
    assert result.returncode == 0, text
    short_sha = hook.git("rev-parse", "--short", "HEAD")
    assert f"checking dev at {short_sha} (fast)" in text
    assert "in container sonitra-devcontainer-1" in text
    args = hook.exec_args()
    assert flag_value(args, "-u") == "node"
    assert flag_value(args, "-w") == "/workspace"
    assert "sonitra-devcontainer-1" in args
    assert args[-3:-1] == ["bash", "-lc"], args
    assert "check.sh" in args[-1]
    assert "--full" not in args[-1]


def test_docker_exec_does_not_let_msys_rewrite_its_arguments(hook: Hook) -> None:
    # What keeps pushes working on Git for Windows: its bundled MSYS bash would
    # rewrite `-w /workspace` before the native docker.exe sees it, and the
    # daemon then rejects the path it was handed. The fix is a command
    # assignment, invisible in argv, so the assertions above cannot catch it.
    hook.set_container_names("sonitra-devcontainer-1")
    result = hook.push(hook.branch_line("dev"))
    assert result.returncode == 0, output(result)
    assert "exec=1" in hook.env_log(), hook.env_log()


def test_failing_checks_fail_the_push(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    hook.set_exec_exit(1)
    result = hook.push(hook.branch_line("dev"))
    assert result.returncode == 1, output(result)
    assert hook.calls("exec")


def test_main_at_head_runs_the_full_checks(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    hook.checkout("main")
    result = hook.push(hook.branch_line("main"))
    assert result.returncode == 0, output(result)
    assert "(full)" in output(result)
    assert "--full" in hook.exec_args()[-1]
    assert "check.sh" in hook.exec_args()[-1]


def test_configured_container_is_used_without_docker_ps(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1", "sonitra-devcontainer-2")
    hook.set_container_config("custom")
    result = hook.push(hook.branch_line("dev"))
    assert result.returncode == 0, output(result)
    assert hook.calls("ps") == []
    assert "custom" in hook.exec_args()


# -- refusals ------------------------------------------------------------------


def test_a_stopped_devcontainer_is_refused(hook: Hook) -> None:
    hook.set_container_names()
    text = assert_refused(hook.push(hook.branch_line("dev")))
    assert "not running" in text
    assert hook.calls("exec") == []


def test_several_running_devcontainers_are_refused(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1", "sonitra-devcontainer-2")
    text = assert_refused(hook.push(hook.branch_line("dev")))
    assert "sonitra.checkContainer" in text
    assert hook.calls("exec") == []


def test_a_failing_docker_ps_is_refused(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    hook.set_ps_exit(1)
    text = assert_refused(hook.push(hook.branch_line("dev")))
    assert "docker ps" in text
    assert hook.calls("exec") == []


def test_a_missing_docker_is_refused(hook: Hook) -> None:
    text = assert_refused(
        hook.push(hook.branch_line("dev"), docker_on_path=False)
    )
    assert "docker" in text and "PATH" in text
    assert hook.calls() == []


def test_pushing_a_branch_that_is_not_checked_out_is_refused(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    hook.checkout("main")
    text = assert_refused(hook.push(hook.branch_line("dev")))
    assert "check out dev before pushing it" in text
    assert hook.calls() == []


def test_a_dirty_working_tree_is_refused(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    hook.make_dirty()
    text = assert_refused(hook.push(hook.branch_line("dev")))
    assert "commit or stash your changes first" in text
    assert hook.calls() == []


def test_pushing_dev_and_main_from_one_checkout_is_refused(hook: Hook) -> None:
    hook.set_container_names("sonitra-devcontainer-1")
    text = assert_refused(hook.push(hook.branch_line("dev"), hook.branch_line("main")))
    assert "check out " in text
    assert hook.calls() == []
