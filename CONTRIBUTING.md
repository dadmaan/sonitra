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

## PR Expectations

- Keep PRs reviewable by splitting broad work into logical commits (e.g. new backend vs. its tests vs. config schema change).
- New config fields must be added to the `PipelineConfig` Pydantic tree with `extra="forbid"` maintained on the affected section.
