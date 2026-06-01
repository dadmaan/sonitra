# Contributing

Thanks for contributing to sonitra.

## Commit Message Convention

Use this format:

`<type>(<scope>): <short imperative summary>`

- Example: `fix(ghsom): honor explicit CLI overrides in sweep agent`
- Keep subject lines concise (target <= 72 characters)
- Use imperative verbs (`add`, `fix`, `refactor`, `update`)
- Prefer one clear scope (`ghsom`, `preprocessing`, `training`, `inference`, `evaluation`, `analysis`, `curriculum`, `networks`, `agents`, `scripts`, `utils`, `configs`, `benchmark`, `tui`, `docs`, `tests`)
- If truly cross-cutting, use `core`
- Recommended types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`, `revert`
- Add a short body for non-trivial changes to explain why and any contract/config impact
- Add `BREAKING CHANGE:` footer when behavior or interfaces are not backward compatible

## Local Quality Workflow

Before opening a PR for non-trivial changes, run:

```bash
pip install -e ".[dev]"
ruff check src/ tests/ scripts/
ruff format src/ scripts/
pytest
```

If your change affects docs surfaces (`docs/**`, `.readthedocs.yaml`, `.github/workflows/docs.yml`, `docs/conf.py`) or API docstrings in `src/**`, also run:

```bash
cd docs
LC_ALL=C.UTF-8 LANG=C.UTF-8 make html
LC_ALL=C.UTF-8 LANG=C.UTF-8 make strict
```

## Pre-commit Hooks

Phase 5 introduces a staged baseline `.pre-commit-config.yaml` focused on active docs/versioning and policy-diagnosis quality surfaces. Expand scope incrementally as legacy formatting/lint debt is reduced.

Install once per clone:

```bash
pip install pre-commit
pre-commit install
```

Run before pushing broad/cross-cutting changes:

```bash
pre-commit run --all-files
```

## Docstring and Docs Enforcement (Staged Ratchet)

ARIA uses staged enforcement rather than a one-shot project-wide strict jump:

1. New/changed files must not introduce new docs build warnings.
2. Stage A: enforce docstring checks on changed files only.
3. Stage B (current): enforce `ruff check --select D --ignore D1` on high-value paths:
   - `src/aria/analysis/policy_diagnosis/`
   - `src/aria/utils/experiment_loading/`
   - `scripts/evaluation/run_mir_evaluation.py`
   - `scripts/benchmark/run_policy_diagnosis_batch.py`
4. Stage C: broaden coverage based on cleanup backlog and repository readiness.

The staged policy is additive: never enable project-wide D-rules in one pass.

## Changelog and Versioning Expectations

For user-visible changes (features, fixes, behavior shifts, docs workflow policy updates), update `CHANGELOG.md` under `## [Unreleased]`.

- Prefer `Added` / `Changed` / `Fixed` categories.
- Explain impact/contract changes briefly.
- Version bumps and release tags are release-governance actions, not routine feature-PR actions.

## Branch and PR Expectations

- Default integration target is `dev` unless a scoped execution plan specifies a different branch.
- For plan-driven work, follow the phase order and branch workflow defined in the plan artifacts.
- Keep PRs reviewable by splitting broad work into logical commits (for example, API doc cleanup vs CI policy changes).
