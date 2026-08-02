# AgentNet Documentation

AgentNet is a Python framework for trainable networks of ReAct agents.

## Project Status

The project has completed Milestone 17 documentation for the current v0.1 API
surface. Public APIs remain pre-1.0, but the guides below cover the implemented
runtime, training, topology search, distributed execution, MCP integration,
tracing, package export, and extension points.

## Guides

- [Quickstart](quickstart.md): build and run a minimal network, save an
  artifact, and use the CLI.
- [Architecture Guide](architecture.md): understand agents, graph containers,
  runtime state, dependency injection, and descriptor-safe artifacts.
- [Training Guide](training.md): fit candidate networks with datasets,
  objectives, history, budgets, and checkpoints.
- [Topology Search Guide](topology-search.md): generate bounded architecture
  mutations and score topology candidates.
- [Distributed Runtime Guide](distributed-runtime.md): run nodes through local,
  thread, process, and injected remote schedulers.
- [MCP Guide](mcp.md): register MCP descriptors, allow tools, adapt them into
  AgentNet tools, and preserve approval controls.
- [LangSmith Guide](langsmith.md): collect normalized traces and export them
  through an injected LangSmith-compatible client.
- [Package Export Guide](package-export.md): export validated `.agentnet`
  artifacts as installable Python packages.
- [Plugin Guide](plugins.md): use the Milestone 19 plugin manager, category
  registries, and descriptor-only extension points.
- [Enterprise Guide](enterprise.md): apply secret handling, approval, validation,
  observability, and deployment practices.
- [API Reference](api-reference.md): review the public v0.1 symbols exported
  from `agentnet`.
- [Release Candidate](release-candidate.md): run the Milestone 20 validation,
  benchmark, build, and credential-gated publish checks.
- [GitHub Publishing](github-publishing.md): create, authenticate, push, and
  troubleshoot the private GitHub repository.

## Local Validation

```bash
uv run ruff check .
uv run pyright
uv run pytest
uv run python benchmarks/release_candidate.py
uv build
```
