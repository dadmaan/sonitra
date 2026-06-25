# Contributing

Thanks for contributing to Sonitra.

## Commit Message Convention

Use this format:

`<type>(<scope>): <short imperative summary>`

- Example: `fix(evaluation): skip NaN-only rows in metric aggregation`
- Keep subject lines concise (target <= 72 characters)
- Use imperative verbs (`add`, `fix`, `refactor`, `update`)
- Prefer one clear scope:
  - `pipeline`, `synth`, `effects`, `separation`, `transcribe`, `evaluation`
  - `benchmark`, `api`, `cli`, `config`, `scripts`, `tests`, `docs`, `ci`, `chore`
  - If truly cross-cutting, use `core`
- Recommended types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`, `revert`
- Add a short body for non-trivial changes to explain why and any contract/config impact
- Add `BREAKING CHANGE:` footer when behaviour or interfaces are not backward compatible

## Local Quality Workflow

`pytest` is the quality gate — there is no ruff or mypy config in this project.

Before opening a PR for non-trivial changes:

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

Sonitra keeps one changelog file per release under `CHANGELOG/`. For user-visible changes (features, fixes, behaviour shifts), add an entry to the appropriate file or create a new one for an upcoming version.

- Use `Added` / `Changed` / `Fixed` categories.
- Briefly explain impact and any contract/config changes.
- Version bumps and release tags are release-governance actions, not routine PR actions.

## Branch and PR Expectations

- Default integration target is `main`.
- Keep PRs reviewable by splitting broad work into logical commits (e.g. new backend vs. its tests vs. config schema change).
- New config fields must be added to the `PipelineConfig` Pydantic tree with `extra="forbid"` maintained on the affected section.
