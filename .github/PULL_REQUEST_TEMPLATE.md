## Description

Provide a clear summary of what changes this PR makes. Reference any related issues (e.g., `Fixes #123`).

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that alters existing behavior)
- [ ] Documentation update
- [ ] Code style or refactoring (no functional change)
- [ ] Test addition or improvement
- [ ] Dependency update

## Affected Component(s)

- [ ] `/mod`
- [ ] `/nn`
- [ ] `/rl`
- [ ] `/bot` (if you have access)
- [ ] Root-level documentation or config

## Testing

Describe how you've tested these changes:

- [ ] Manual testing (describe steps)
- [ ] Unit tests added/updated
- [ ] Smoke test passes (for `/nn`)
- [ ] Full test suite passes locally

**For `/mod` changes:** Have you tested on a real Minecraft client?

**For `/nn` changes:** Does `pytest` pass and does `python -m herbert_nn.smoketest` complete successfully?

**For `/rl` changes:** Describe your testing environment.

## Code Quality

- [ ] Code follows the style guide (Google Java Style, Black/ruff/mypy for Python)
- [ ] All public APIs have complete docstrings/Javadoc/JSDoc
- [ ] No TODOs, FIXMEs, or commented-out code added
- [ ] No magic numbers (all extracted to named constants with comments)
- [ ] No hardcoded secrets or credentials
- [ ] `.gitignore` remains accurate (no new untracked files accidentally committed)

## Documentation

- [ ] Updated relevant README.md(s) if behavior changed
- [ ] Added docstrings to new public functions/classes
- [ ] Updated CHANGELOG.md if this is a user-facing change

## Breaking Changes

Does this PR introduce a breaking change to any of the data contracts?

- [ ] JSONL schema changes (if so, explain and update `mod/README.md`)
- [ ] Discord webhook upload format changes
- [ ] `/nn` checkpoint loading changes
- [ ] `/rl` environment contract changes

If yes, describe the change and migration path for users.

## Additional Notes

Any other context or concerns? Flag if this PR blocks or is blocked by other work.
