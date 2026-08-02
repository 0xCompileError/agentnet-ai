# Contributing

Keep changes small, tested, and aligned with the issue or pull request they
address.

## Local Setup

Install development dependencies:

```bash
uv sync --dev
```

## Validation

Run the same checks used by CI before sending changes:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

## Pull Requests

- Keep each pull request scoped to one task.
- Add or update tests with behavior changes.
- Update user-facing documentation and the changelog when behavior changes.
