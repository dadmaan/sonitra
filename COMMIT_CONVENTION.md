## Commit Message Convention

Use this format:

`<type>(<scope>): <short imperative summary>`

- Example: `feat(api): add health check endpoint`
- Keep subject lines concise (target <= 72 characters)
- Use imperative verbs (`add`, `fix`, `refactor`, `update`)
- Prefer one clear scope (`api`, `frontend`, `devcontainer`, `cli`, `docs`, `tests`, `scripts`, `configs`, `core`)
- If truly cross-cutting, use `core`
- Recommended types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`, `revert`
- Add a short body for non-trivial changes to explain why and any contract/config impact
- Add `BREAKING CHANGE:` footer when behavior or interfaces are not backward compatible
