# Architecture Guide

AgentNet is organized around a small set of explicit abstractions:
`ReActAgent` nodes, Graph containers, runtime context, dependency injection, and
descriptor-safe artifacts.

## Core Layers

- `Module` is the base executable unit. Every graph and agent is a module.
- `ReActAgent` is the default agent node. It owns instructions, LLM choices,
  allowed tool names, retry policy, interfaces, and metadata.
- Graph containers are modules that coordinate other modules: `Sequential`,
  `Parallel`, `Router`, `Reducer`, and `DAG`.
- `RunContext` carries `GraphState`, cancellation state, metadata, retry events,
  tool events, MCP events, scheduler events, and tracing data through a run.
- Artifact APIs save descriptors and require live dependencies at load time.

## Agents

`ReActAgent` receives one or more LLM backends or model aliases. Live backends
execute immediately. Aliases are for serialization and must be mapped back to
live backends when loading an artifact.

```python
import agentnet as an

agent = an.ReActAgent(
    "researcher",
    instructions="Find the minimum relevant evidence.",
    llms=["strong", "cheap"],
    tools=["search_docs"],
    retry_policy=an.RetryPolicy(
        transport_retries=2,
        fallback_on=("api_error", "timeout"),
    ),
)
```

## Graph Containers

Graph containers make architectures explicit and testable:

```python
import agentnet as an

pipeline = an.Sequential(
    an.ReActAgent("planner", llms=["cheap"], instructions="Plan."),
    an.ReActAgent("critic", llms=["strong"], instructions="Find risks."),
    an.ReActAgent("writer", llms=["cheap"], instructions="Write."),
    name="review_pipeline",
)

compiled = an.compile_graph(pipeline)
print(compiled.entry_nodes)
print(an.visualize_graph(pipeline))
```

Use `GraphValidator` or `validate_graph` before persisting or optimizing a graph
when the graph is assembled dynamically.

## Dependency Injection

Dependency injection is the production boundary. Artifacts reference model
aliases and tool names, but the application supplies live objects:

```python
loaded = an.load(
    "review_pipeline.agentnet",
    llms={
        "cheap": an.FakeLLM(["draft"], name="cheap"),
        "strong": an.FakeLLM(["critique"], name="strong"),
    },
    tools={
        "search_docs": lambda query: [query],
    },
)
```

This keeps infrastructure, credentials, private URLs, clients, and executable
tool code outside serialized artifacts.

## Runtime State

`RunContext` is optional for simple runs, but pass one when collecting metadata
or sharing cancellation state:

```python
import agentnet as an

context = an.RunContext(run_id="run-123")
tracer = an.InMemoryTracer()

an.run(
    an.ReActAgent("planner", llms=[an.FakeLLM(["ok"], name="strong")]),
    "input",
    context=context,
    tracer=tracer,
)

trace = an.trace_from_context(context)
print(trace.metrics.to_dict())
```

## Serialization Boundary

`.agentnet` artifacts serialize descriptors only. They can contain graph
structure, prompts, schemas, model aliases, retry policy descriptors, tool
manifests, MCP manifests, and training history. They do not serialize live LLM
clients, tool implementations, MCP clients, scheduler clients, or credentials.
