# AgentNet

AgentNet is a Python framework for building, evaluating, training, serializing,
and deploying networks of ReAct agents.

The core idea is to treat an agent system like a model architecture: define
agents as executable modules, compose them into explicit graphs, optimize their
prompts and policies against objectives, then ship descriptor-only artifacts
that are loaded with production dependencies at runtime.

AgentNet is currently a v0.1 release-candidate codebase. The public API is
pre-1.0, but the repository already includes the runtime, graph system, LLM
layer, tools, MCP integration, evaluation, training, topology search,
serialization, package export, tracing, plugins, CLI, documentation, examples,
and tests for the implemented surface.

## Why AgentNet

Most agent frameworks focus on orchestration. AgentNet focuses on trainable
agent networks:

- Define architectures with Python classes such as `ReActAgent`, `Sequential`,
  `Parallel`, `Router`, `Reducer`, and `DAG`.
- Configure LLMs explicitly in code, including per-agent model fallback
  ordering.
- Attach tool allowlists per agent and execute tools only through explicit
  registries.
- Validate interfaces, representations, constraints, schemas, retry policies,
  costs, latency, tools, topology, memory, and safety.
- Evaluate and train candidate networks with objectives, datasets, budgets,
  checkpoints, attribution, and reversible patches.
- Train directly from ordinary Python inputs and expected outputs with
  `trained = agentnet.train(net, X, y)`.
- Search bounded topology mutations rather than freezing architecture decisions
  by hand.
- Save `.agentnet` artifacts that store descriptors, not secrets or executable
  runtime objects.
- Export validated artifacts as installable Python packages.
- Collect normalized traces and export them through injected LangSmith or
  OpenTelemetry-compatible clients.

## What AgentNet Provides

### Runtime Core

- `Module` as the base executable unit.
- Sync and async execution through `agentnet.run` and `agentnet.arun`.
- `RunContext`, `GraphState`, `AgentState`, and `GraphResult` for runtime state
  and metadata.
- Cancellation support, retry events, scheduler events, tracing metadata, and a
  public runtime error hierarchy.

### Agents And Graphs

- `ReActAgent` with instructions, LLM policy, retry policy, tool whitelist,
  input validation, output interface validation, and max-step protection.
- Graph containers: `Sequential`, `Parallel`, `Router`, `Reducer`, and `DAG`.
- Shape DSL construction through `build_shape`.
- Graph compilation, validation, and Mermaid visualization helpers.

### LLMs, Tools, And MCP

- Provider protocol: `LLMBackend`.
- Built-in backends: `FakeLLM`, `LiteLLM`, `OpenAI`, `Anthropic`, `Bedrock`,
  `VertexAI`, and `OpenAICompatible`.
- `ChatRequest`, `ChatResponse`, and `ChatEvent` records, including streaming
  support.
- `ToolSpec` and `ToolRegistry` for explicit in-process tools.
- MCP descriptor support through `MCPServer`, `MCPToolDescriptor`,
  `MCPRegistry`, `MCPApprovalStore`, `MCPToolAdapter`, and `FakeMCPServer`.

### Evaluation, Training, And Optimization

- Objective framework with schema, judge, exact-match, human-feedback,
  unit-test, cost, latency, tool-efficiency, and custom objectives.
- `Trainer`, `Dataset`, `TrainingExample`, `TrainingHistory`,
  `TrainingCheckpoint`, and `BudgetManager`.
- Prompt, fallback, retry-policy, representation-selection,
  communication-protocol, translation-strategy, information-transfer, interface
  compatibility, and topology optimizers.
- Constraint-aware candidate selection and bounded topology search.

### Serialization, Export, And Observability

- Descriptor-only `.agentnet` artifact save, load, validation, manifest, schema,
  tool, MCP, prompt, and training-history support.
- Package export for validated artifacts with generated loaders and project
  metadata.
- `TraceEvent`, `TraceSpan`, `Trace`, `TraceMetrics`, `InMemoryTracer`,
  `trace_from_context`, `LangSmithExporter`, and `OpenTelemetryExporter`.
- Plugin registries for providers, optimizers, evaluators, schedulers, tracers,
  storage, and memory.

## Quickstart

Install development dependencies with `uv`:

```bash
uv sync --dev
```

Run a single deterministic agent:

```python
import agentnet as an

planner = an.ReActAgent(
    "planner",
    instructions="Return a concise migration plan.",
    llms=[an.FakeLLM(responses=["Inventory dependencies, test, then cut over."], name="strong")],
)

answer = an.run(planner, "Move the reporting job to the new warehouse.")
print(answer)
```

Compose agents into a graph:

```python
import agentnet as an

net = an.Sequential(
    an.ReActAgent(
        "planner",
        instructions="Break the request into steps.",
        llms=[an.FakeLLM(["Plan the analysis."], name="planner-model")],
    ),
    an.ReActAgent(
        "writer",
        instructions="Write the final response.",
        llms=[an.FakeLLM(["Final response."], name="writer-model")],
    ),
    name="decision_net",
)

print(an.run(net, "Should we move the batch job?"))
print(an.visualize_graph(net))
```

Train from ordinary Python data and immediately use the fitted network:

```python
import agentnet as an

classifier = an.ReActAgent(
    "classifier",
    instructions="Return only billing or technical.",
    llms=[
        an.FakeLLM(
            ["billing", "technical", "billing", "technical", "billing"] * 3
        )
    ],
)

trained = an.train(
    classifier,
    ["wrong invoice", "server error", "refund", "timeout", "charged twice"],
    ["billing", "technical", "billing", "technical", "billing"],
)

print(trained.training.summary())
print(trained.run("new support request"))
```

`train` infers expected-output scoring, creates a deterministic validation split,
uses bounded optimization, and returns a runnable network. Objectives, optimizers,
validation data, proposal models, and budgets remain optional advanced controls.

Save a descriptor-only artifact and load it with explicit runtime dependencies:

```python
import agentnet as an

net = an.ReActAgent(
    "planner",
    instructions="Plan clearly.",
    llms=["strong"],
)

an.save(net, "planner.agentnet", name="planner")

loaded = an.load(
    "planner.agentnet",
    llms={"strong": an.FakeLLM(responses=["loaded"], name="strong")},
)

print(an.run(loaded, "Use the saved artifact."))
```

Inspect and export artifacts with the CLI:

```bash
uv run agentnet --help
uv run agentnet inspect planner.agentnet
uv run agentnet export planner.agentnet --package planner-net --output dist/planner-net
```

## Safety And Serialization Model

AgentNet keeps serialized artifacts declarative. Artifacts may include graph
structure, prompts, schemas, model aliases, retry policy descriptors, tool
descriptors, MCP descriptors, constraints, training history, metrics, and
version metadata.

Artifacts do not serialize:

- API keys, tokens, private headers, or environment variables.
- Live LLM clients, MCP clients, schedulers, or tracing clients.
- Tool implementations or arbitrary Python callables.
- Dynamically loaded plugin code.

Production applications inject live dependencies when loading artifacts. This
keeps infrastructure and credentials outside the artifact boundary and makes the
same artifact usable across local tests, staging, and production.

## Examples

Runnable examples live under `examples/`:

- `01_single_agent`
- `02_sequential_pipeline`
- `03_parallel_research_pipeline`
- `04_router`
- `05_mixture_of_experts`
- `06_litellm_gateway`
- `07_mcp`
- `08_langsmith`
- `09_distributed_execution`
- `10_topology_search`
- `11_exported_package`
- `12_training_10_examples`

Run the example coverage with:

```bash
uv run pytest tests/test_examples.py
```

## Documentation

Project documentation lives under `docs/` and is wired through `mkdocs.yml`.
Start with:

- [Quickstart](docs/quickstart.md)
- [Architecture Guide](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Topology Search Guide](docs/topology-search.md)
- [Distributed Runtime Guide](docs/distributed-runtime.md)
- [MCP Guide](docs/mcp.md)
- [LangSmith Guide](docs/langsmith.md)
- [Package Export Guide](docs/package-export.md)
- [Plugin Guide](docs/plugins.md)
- [Enterprise Guide](docs/enterprise.md)
- [API Reference](docs/api-reference.md)
- [Release Candidate Runbook](docs/release-candidate.md)

## Development Workflow

Run the standard validation suite before merging changes:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

Build release artifacts locally with:

```bash
uv build
```

The repository uses Hatchling, Ruff, Pyright, pytest, coverage, pre-commit, and
GitHub Actions CI.

## Release Status

The source tree declares version `0.1.0`. The release candidate has been built,
validated, uploaded to TestPyPI, and verified from the TestPyPI wheel.

Production PyPI publishing is intentionally blocked until project ownership or a
package naming/versioning decision is resolved. The existing PyPI `agentnet`
project already has a `0.1` release, which normalizes to `0.1.0`.

## License

AgentNet is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
