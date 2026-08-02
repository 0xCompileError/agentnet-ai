# Release Candidate Runbook

Milestone 20 verifies that AgentNet v0.1.0 is locally shippable before any
public release. The local checks are deterministic and do not require service
credentials. TestPyPI and PyPI uploads are credential-gated and must use
tokens or trusted publishing outside the repository.

## Local Validation

Run these commands from the repository root:

```bash
uv run ruff check .
uv run pyright
uv run pytest
uv run pytest tests/test_artifacts_serialization.py tests/test_package_export.py
uv run pytest tests/test_mcp_integration.py tests/test_tracing_observability.py
uv run pytest tests/test_scheduler_runtime.py tests/test_examples.py
uv run python benchmarks/release_candidate.py
uv build
```

The full `uv run pytest` command is the unit-test gate. The focused pytest
commands cover integration and end-to-end release surfaces:

- serialization compatibility through `.agentnet` save, load, and validation;
- package export through generated package loaders;
- LangSmith integration through an injected compatible client;
- MCP integration through fake MCP descriptors, approval, allowlists, and tool
  adaptation;
- distributed runtime through local scheduler abstractions;
- examples as end-to-end executable workflows.

The benchmark suite is currently `benchmarks/release_candidate.py`. It uses
`FakeLLM` and the public runtime API so it stays deterministic and does not
contact external services.

## Build Artifacts

Build source and wheel distributions with:

```bash
uv build
```

Expected artifacts are written under `dist/` and should include a source
distribution and a universal wheel for `agentnet` version `0.1.0`.

## TestPyPI Publish

Use TestPyPI for the release-candidate upload. Do not store credentials in
source files, generated artifacts, docs, or chat logs.

Recommended local pattern on macOS:

```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="$(security find-generic-password -a agentnet -s agentnet-testpypi-token -w)"
uv run --with twine twine upload --repository-url https://test.pypi.org/legacy/ dist/*
unset TWINE_PASSWORD
```

After upload, verify installation from TestPyPI in a clean environment before
publishing to PyPI.

## PyPI Publish

The production PyPI publish requires separate explicit approval and production
credentials or trusted publishing configuration. Do not reuse TestPyPI
credentials for PyPI.

Preferred production release flow:

```bash
uv run ruff check .
uv run pyright
uv run pytest
uv build
```

Then publish with trusted publishing from CI, or with a project-scoped token
retrieved from a secure local store. Production upload should happen only after
TestPyPI installation has been verified.
