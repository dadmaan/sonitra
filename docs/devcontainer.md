# Devcontainer

The devcontainer gives you a ready Linux environment for Sonitra. It runs in Docker and opens in VS Code, or headless with the devcontainer CLI. You do not need to install Python, uv, or system tools on your machine. The setup lives in `.devcontainer/` (`devcontainer.json`, the `Dockerfile`, `post-create.sh`), and the repo appears at `/workspace` inside the container.

## Rebuild the container

The rebuild must go through the devcontainer tooling, not plain Docker Compose.

In VS Code: Command Palette → **Dev Containers: Rebuild Container**.

Headless, with the devcontainer CLI:

```bash
npx -y @devcontainers/cli up --workspace-folder . --remove-existing-container
```

Do not use `docker compose build` in `.devcontainer/` as a rebuild path. It updates an image tag but does not recreate the container your editor runs. Rebuild through VS Code or the CLI, then check the running container's image ID if you need to confirm it: image names differ between tool versions.

## What the create step does

`postCreateCommand` in `.devcontainer/devcontainer.json` runs `.devcontainer/post-create.sh`. The script picks the lock's torch build for the image and syncs it into `/workspace/.venv`:

- CPU fork: `uv sync --locked --extra transkun --extra dev`
- GPU fork: `uv sync --locked --extra transkun-gpu --extra xla-ptx --extra dev`

An extra is a named group of optional dependencies, such as `transkun` and `dev`. A fork here is one of the lock's two torch builds, CPU or CUDA. The GPU fork carries the narrow `xla-ptx` extra as well: `nvidia-cuda-nvcc-cu12` ships `ptxas` and `nvvm/libdevice` for TensorFlow 2.15's runtime PTX codegen. TF 2.15 wheels carry cubins only to sm_75 (compute_80 PTX above that), so on newer GPUs every kernel_gen/XLA op is JIT-compiled at runtime and needs libdevice from the same environment as TensorFlow — `LD_LIBRARY_PATH` cannot supply it.

The fork follows the image, not your host. The script checks for `$NV/cudnn/lib`, a folder that exists only when the image was built with the GPU override (`docker-compose.gpu.yml`). A GPU host that starts the base compose file gets the CPU fork, which matches the container's absent GPU passthrough.

Re-run the script at any time:

```bash
bash .devcontainer/post-create.sh
```

To see which fork it would pick, without syncing anything:

```bash
bash .devcontainer/post-create.sh --print-extra
```

## Two Python environments

- `/workspace/.venv` is the main environment (an isolated Python folder). Bare `sonitra` and `python` run from `/workspace/.venv/bin` because that folder comes first on `PATH` (the list of folders your shell searches for commands). It holds torch and transkun on the torch build the lock names, plus `ptxas`/`libdevice` via `xla-ptx` on the GPU fork. VS Code uses it too (`/workspace/.venv/bin/python`). It lives on the bind mount (a shared folder between your machine and the container), so it survives a rebuild. This takes effect on Rebuild Container. The venv also wins in login bash: the image appends the same prepend to `/home/node/.profile`, after Debian's stock `$HOME/.local/bin` block.
- The pip `--user` site is the image's own environment (`/usr/local/bin/python`). It has the core stack (TensorFlow, basic-pitch, Jupyter, R) but no transkun and no torch. It works even when the sync fails.

`uv run --no-sync <command>` still works. It runs the venv without changing anything, so it is the mutation-free form. You can also call `/workspace/.venv/bin/python` directly. Both leave the environment alone.

`pip` is not in the venv. uv venvs ship with no pip, so bare `pip list` still reports the image environment even after the PATH fix. Use `uv pip list` or `python -m pip list` to see what the venv holds.

## The `uv sync` footgun

`uv sync` is exact by default. It removes anything the requested set does not name, and extras are not in the default set. A bare `uv sync --locked` in this container would uninstall 53 packages, transkun, torch, and pytest among them.

Safe ways to sync:

- Run the script: `bash .devcontainer/post-create.sh`.
- Run it by hand with the extras: `uv sync --locked --extra transkun --extra dev` (GPU: `--extra transkun-gpu --extra xla-ptx --extra dev`).
- `uv run` is inexact. It installs what the default set needs and removes nothing, so `uv run python -m pytest` is safe. `uv run --no-sync` skips the check and mutates nothing. Avoid `uv run --exact`, which strips the extras.

### Switching forks needs `--reinstall-package`

A plain `uv sync` cannot move an existing venv from the CPU fork to the GPU fork. Both forks pin the same version (`torch==2.12.1`); only the build differs. The CPU wheel carries the `+cpu` local tag, and the PyPI CUDA wheel does not. A requirement with no local tag accepts any local tag, so `2.12.1+cpu` satisfies `==2.12.1`. uv checks the environment, decides it is already correct, prints `Would make no changes`, and leaves the CPU torch in place.

The reverse direction is different. The CPU fork pins `torch==2.12.1+cpu`, which a CUDA build does not satisfy, so switching GPU to CPU normally swaps on its own.

To switch by hand, force the swap:

```bash
# CPU fork
uv sync --locked --extra transkun --extra dev \
    --reinstall-package torch --reinstall-package torchaudio

# GPU fork
uv sync --locked --extra transkun-gpu --extra xla-ptx --extra dev \
    --reinstall-package torch --reinstall-package torchaudio
```

A wrong build fails the preflight check with:

```
transkun device 'cuda' resolved to 'cuda' but the installed torch (2.12.1+cpu) is a CPU-only build. Reinstall the CUDA fork: `uv sync --locked --extra transkun-gpu --extra xla-ptx --extra dev --reinstall-package torch --reinstall-package torchaudio` (a plain sync will not swap the build: both forks pin the same version).
```

`torch.__version__` ends in `+cpu` when this fires. The older message, `transkun device 'cuda' resolved to 'cuda' but CUDA is not available`, now covers a different case: a real CUDA build that cannot see a GPU (a passthrough problem), not a wrong build.

Check which build is installed:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`torch.version.cuda` is `None` on a CPU build and a version string (e.g. `13.0`) on a CUDA build.

`.devcontainer/post-create.sh` detects a wrong build and re-runs the sync with these flags itself. The manual command above is therefore only needed for a hand-run `uv sync`. The venv at `/workspace/.venv` lives on the bind mount, so a rebuild does not repair a wrong build on its own: the create step re-runs the same lock and audits clean again.

## Run the tests

```bash
# Heavy backends (basic-pitch, transkun). On the GPU devcontainer: 5 passed, 1 skipped
# (the skip is test_transkun_cuda_unavailable_raises, which skips when CUDA is present).
uv run --no-sync python -m pytest tests/test_transkun.py -m slow -q

# Everything else. Measured: 988 passed, 12 skipped, 42 deselected, 0 failures.
uv run --no-sync python -m pytest tests/ -m "not slow" -q
```

On a CPU devcontainer, the slow command gives 6 passed.

## Notes

- The rebuilt image is 7.93 GB. uv adds about 43 MB.
- The venv lives on the `/workspace` mount and persists across rebuilds. The uv cache does not: it lives in the container layer and is lost on rebuild.
- uv prints a hardlink warning during the sync because the cache and the venv are on different filesystems. The warning is harmless. Set `UV_LINK_MODE=copy` to silence it, at the cost of slower, larger writes.
- On Windows, the devcontainer mounts the repo from the Windows filesystem through WSL2. That mount can make `uv sync` fail with I/O errors on deep file trees. See the WSL2 note in [README.md](../README.md).
- A fresh venv sync downloads about 4.0 GB on the CPU fork and 6.9 GB on the GPU fork. A re-sync against an existing venv is small, except a fork switch, which reinstalls torch and torchaudio and downloads about 3 GB.

## Troubleshooting

- The create step failed, or you are offline: the pip site still works, so you can keep editing and run commands that do not need transkun. When the network returns, re-run `bash .devcontainer/post-create.sh` and check the Dev Containers log.
- `transkun` is missing from the venv: check `bash .devcontainer/post-create.sh --print-extra`, then re-run the script.
- `uv sync` reports `/workspace/.venv` is not a valid Python environment (no Python executable found): the bind-mounted venv is a broken stub, usually from an interrupted sync. The script now removes and recreates it automatically; if the container never started, delete `.venv` on the host and rebuild.
