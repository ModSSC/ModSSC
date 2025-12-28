# Governance

## Roles
Maintainers:
- Review and merge pull requests
- Cut releases and manage versioning
- Decide on architectural changes

Contributors:
- Propose changes via pull requests
- Add tests, docs, and changelog fragments

## Decision process
- Small changes: maintainer review + CI green
- Architectural changes: short design note in the PR description, maintainer approval

## Quality gates for new methods
A new method must include:
- tests (at least smoke + unit)
- a quickstart example
- a recipe / config used for reproduction or benchmark
- a changelog fragment
