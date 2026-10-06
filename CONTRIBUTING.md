# Contributing

Thanks for contributing to Sonitra.

## Commit Messages

Follow [`COMMIT_CONVENTION.md`](COMMIT_CONVENTION.md): `<type>(<scope>): <short imperative summary>`.

## Local Quality Workflow

`pytest` is the quality gate — there is no ruff or mypy config in this project.

Before opening a PR for non-trivial changes:

```bash
uv sync --extra dev     # recommended: install with dev deps (uses lockfile)
uv run pytest           # run the full suite
```

If uv is unavailable, fall back to pip:

```bash
pip install -e ".[dev]"
pytest
```

To skip slow tests (Basic Pitch TF inference) during iteration:

```bash
pytest -m "not slow"
```

To skip tests that require a VST plugin:

```bash
pytest -m "not skip_if_no_vst"
```

## Changelog

Sonitra keeps a single changelog at `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/) format. For every user-visible change (feature, fix, behaviour shift), add an entry under the `[Unreleased]` section using the `Added` / `Changed` / `Fixed` categories.

- Each entry should briefly explain the impact and any contract/config changes.
- When a new version is released, the `[Unreleased]` heading is renamed to the version number and dated; an empty `[Unreleased]` section is added for the next cycle.

## Branches

Sonitra uses two long-lived branches:

| Branch | Purpose | Stability |
|--------|---------|-----------|
| `main` | Stable code. Releases are cut from here. | Tests pass. Config schema and CLI are stable between releases. |
| `dev`  | Integration branch for in-progress features and experimental changes. | May be unstable: interfaces, config keys and outputs can change without notice. |

- Open feature and fix PRs against `dev`, not `main`.
- `dev` is merged into `main` once its changes are complete, tested and documented in `CHANGELOG.md`.
- Changelog entries accumulate under `[Unreleased]` on `dev` and reach `main` with the merge.
- Do not depend on `dev` for reproducible benchmark results. Pin to `main` or a release tag.
- Urgent fixes to a release may target `main` directly; they are then merged back into `dev`.

## Checks and releases

Commits happen inside the devcontainer (the containerized development environment), but `git push` runs on the host, your normal machine outside the container. The pre-push hook (the script git runs before a push) runs on the host too. It executes the checks inside the running devcontainer:

```bash
docker exec -u node -w /workspace <container> bash -lc 'bash scripts/ci/check.sh [--full]'
```

It never falls back to testing on the host: if the container cannot be reached, the push is refused.

### One-time hook setup (host, per clone)

```bash
git config core.hooksPath scripts/hooks
```

The hook finds the devcontainer by a Docker Compose label, `com.docker.compose.service=sonitra`. When it does not find exactly one running container, name the container yourself:

```bash
docker ps --filter label=com.docker.compose.service=sonitra --format '{{.Names}}'
git config sonitra.checkContainer <name>
```

`git config` is per clone, not per branch.

### What the hook checks

| Push | Checks |
|--------|---------|
| `dev` | `uv lock --check` plus the fast test suite (`-m "not slow"`) |
| `main` | `uv lock --check` plus the full test suite |
| Tag-only push, branch deletion | Nothing |

The devcontainer must be running to push `dev` or `main`. A `dev` push spends roughly 2-3 minutes in the fast test suite; `main` runs the full suite and takes longer.

The hook also refuses when the working tree is dirty or when `HEAD` is not the commit being pushed. The checks run on the checked-out tree, so the two must match.

Manual runs inside the devcontainer:

```bash
bash scripts/ci/check.sh          # lock check + fast suite
bash scripts/ci/check.sh --full   # lock check + full suite
```

`git push --no-verify` is the emergency bypass. It leaves the push unchecked locally; only the `ci` workflow (fast test suite, clean machine) sees it.

### CI after the push

`.github/workflows/ci.yml` is the post-push safety net. It runs on pushes to `dev` and `main`, on pull requests, and on a manual run from the GitHub Actions tab (`workflow_dispatch`). It installs from a clean machine (`uv sync --locked --extra dev --extra transkun` on Ubuntu, plus the system packages) and runs the same `bash scripts/ci/check.sh`. This catches packages installed by hand in the devcontainer but never declared in `pyproject.toml`, and tests that pass only because of something in the devcontainer image. GPU work, `corpus/`, slow tests, rendering through a VST plugin (a software instrument or effect) and hFT-Transformer model weights stay manual.

### Tags

| Branch | Tag | Meaning |
|--------|-----|---------|
| `main` | `vX.Y.Z` | Stable release |
| `dev` | `vX.Y.Z-dev.N` | Pre-release snapshot (published before the final release) |

Tags are never moved or deleted. A GitHub ruleset (a repository rule) blocks updates and deletions of `v*` tags, and branch rulesets block force-push and deletion on `main` / `dev`.

### Release checklist

1. For a stable release, bump the version in `pyproject.toml` and refresh the lock on the host with `uv lock`. Never commit `uv.lock` from inside the container.
2. Cut the changelog section: in `CHANGELOG.md`, rename `[Unreleased]` to the version and date, then add a fresh empty `[Unreleased]` for the next cycle.
3. Commit the version, lock and changelog changes, then tag and push:

```bash
git tag -a v0.5.0 -m "v0.5.0"
git push origin main
git push origin v0.5.0
```

The GitHub Release then appears by itself. The `release` workflow fires on the `v*` tag and checks the tag against `pyproject.toml` before creating the Release with that version's `CHANGELOG.md` section as notes. A stable tag must match the `pyproject.toml` version exactly. A hyphenated `X.Y.Z-label.N` tag must be a PEP 440 pre-release or dev release of the same base. PEP 440 is Python's version-numbering standard: for example, `v0.5.0-dev.5` matches `0.5.0.dev0`. A hyphenated tag is marked Pre-release on GitHub and may fall back to a line pointing at `CHANGELOG.md` when it has no section of its own. A stable tag without its section fails the workflow. Dev snapshot tags do not edit `pyproject.toml` or refresh the lock.

## PR Expectations

- Keep PRs reviewable by splitting broad work into logical commits (e.g. new backend vs. its tests vs. config schema change).
- New config fields must be added to the `PipelineConfig` Pydantic tree with `extra="forbid"` maintained on the affected section.
