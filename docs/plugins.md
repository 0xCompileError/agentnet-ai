# Plugin Guide

Milestone 19 adds the full plugin manager. AgentNet plugin APIs are explicit and
in-process only: AgentNet records names and descriptors, but does not load
executable code from serialized data.

## Plugin Manager

Use `PluginManager` when an application wants one registry for providers,
optimizers, evaluators, schedulers, tracers, storage, and memory plugins:

```python
import agentnet as an

plugins = an.PluginManager()
plugins.register_provider(
    "fake-llm",
    lambda response="ready": an.FakeLLM([response], name="fake-llm"),
    description="Deterministic test provider.",
)
plugins.register_evaluator(
    "exact",
    lambda expected: an.ExactMatchObjective(expected),
)

llm = plugins.providers.create("fake-llm", response="done")
objective = plugins.evaluators.create("exact", "done")

print(llm.name)
print(objective.evaluate("done").passed)
```

`PluginManager.to_dict()` serializes descriptors only. It does not serialize
factories, callables, clients, credentials, or loaded plugin instances.

## Category Registries

Each category is also available as a standalone registry:

```python
import agentnet as an

providers = an.ProviderPluginRegistry()
providers.register("fake-llm", lambda: an.FakeLLM(["ok"], name="fake-llm"))

storage = an.StoragePluginRegistry()
storage.register("dict-store", dict)

payload = providers.to_dict()
descriptor_only = an.ProviderPluginRegistry.from_dict(payload)
```

Descriptor-only registries preserve plugin metadata for review, packaging, or
policy checks. They cannot create plugin instances until application code
registers factories again.

The typed registries validate known framework contracts:

- `ProviderPluginRegistry` returns an `LLMBackend`.
- `OptimizerPluginRegistry` returns an object with `optimize` or `search`.
- `EvaluatorPluginRegistry` returns an `Objective`.
- `SchedulerPluginRegistry` returns a `Scheduler`.
- `TracerPluginRegistry` returns an object with tracer span methods.
- `StoragePluginRegistry` and `MemoryPluginRegistry` accept application-defined
  objects because storage and memory contracts are intentionally app-specific.

## Constraint Plugins

Use `ConstraintPluginRegistry` when application code needs named factories for
custom constraints.

```python
import agentnet as an


class EqualsConstraint(an.Constraint):
    def __init__(self, expected: object) -> None:
        super().__init__("equals")
        self.expected = expected

    def check(self, candidate: object, context: object | None = None) -> bool:
        return candidate == self.expected


registry = an.ConstraintPluginRegistry()
registry.register("equals", lambda expected: EqualsConstraint(expected))

constraint = registry.create("equals", expected="ready")
print(constraint.evaluate("ready").passed)
```

The registry serializes plugin names only. It does not load executable code from
descriptors.

## Representation Plugins

Use `RepresentationPluginRegistry` for custom interface representations:

```python
import agentnet as an

registry = an.RepresentationPluginRegistry()
registry.register(
    "custom_text",
    lambda: an.PlainTextRepresentation(identifier="custom_text"),
)

representation = registry.create("custom_text")
print(representation.identifier)
```

Pair custom representations with `RepresentationTranslatorRegistry` when values
need explicit conversions between formats.

## Optimizer And Runtime Extension Points

Current extension points are ordinary Python objects:

```python
import agentnet as an

objective = an.CustomObjective(
    "contains-ready",
    lambda output, context: "ready" in str(output),
)

optimizer = an.ConstraintAwareOptimizer(
    objective=lambda candidate: 1.0 if candidate == "ready" else 0.0,
)
```

For schedulers, implement the `Scheduler` protocol or wrap a framework through
an injected client. For tracing, pass a custom LangSmith-compatible client or an
OpenTelemetry-compatible exporter object.

## Safety Rules

Plugin extension points should follow the same repository constraints:

- Register factories explicitly in application code.
- Serialize descriptors, names, and metadata only.
- Do not serialize secrets.
- Do not load executable code from artifacts.
- Record architecture changes in project decisions when extension behavior
  changes public contracts.
