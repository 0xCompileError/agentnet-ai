# Enterprise Guide

AgentNet is designed for code-first configuration, explicit dependency
injection, descriptor-only artifacts, and production controls. The default
enterprise stance is: Do not serialize secrets.

## Secret Handling

Artifacts, traces, checkpoints, MCP descriptors, scheduler metadata, and CLI
reports validate secret-like keys. Keep credentials in the application runtime
or platform secret store, then inject live clients at process startup.

```python
import agentnet as an

net = an.load(
    "support.agentnet",
    llms={"strong": an.FakeLLM(name="strong")},
    tools={"search_docs": lambda query: [query]},
)
```

Do not put API keys, tokens, private headers, or live client objects into
artifact metadata.

## Approval And Allowlists

Use allowlists for tools and MCP descriptors, then add approval for reviewed MCP
tools:

```python
import agentnet as an

registry = an.MCPRegistry()
registry.register(
    an.MCPServer(
        name="ticketing",
        command=["ticketing-mcp"],
        tools=[an.MCPToolDescriptor("create_ticket")],
    ),
    allow_tools=["create_ticket"],
)

approvals = an.MCPApprovalStore()
approvals.approve(
    "ticketing.create_ticket",
    registry.get_tool("ticketing.create_ticket"),
    approved_by="platform",
)
```

Tie approval reviews to descriptor hashes so changed descriptors require a new
review.

## Artifact Validation

Validate artifacts before loading, exporting, or deploying:

```python
import agentnet as an

result = an.validate_artifact(
    "support.agentnet",
    llms={"strong": an.FakeLLM(name="strong")},
    tools={"search_docs": lambda query: [query]},
)

if not result.passed:
    raise RuntimeError(result.failures)
```

Validation catches incompatible artifact versions, hash mismatches, missing LLM
aliases, missing tools, missing MCP allowlists, schema descriptor errors, and
unsafe descriptor payloads.

## Observability

Use normalized traces for audits and operations:

```python
import agentnet as an

context = an.RunContext(run_id="enterprise-run")
tracer = an.InMemoryTracer()

an.run(net, "input", context=context, tracer=tracer)
trace = an.trace_from_context(context)

print(trace.metrics.to_dict())
```

Export traces through injected LangSmith or OpenTelemetry clients. Keep exporter
credentials and endpoint configuration outside descriptors.

## Deployment Checklist

- Validate every artifact in CI and before deployment.
- Require explicit dependency injection for LLMs, tools, MCP clients, schedulers,
  and exporters.
- Keep tool and MCP allowlists narrow.
- Require approval for side-effecting or broad-access tools.
- Use bounded retry policies and topology search budgets.
- Store training history and package export outputs as reviewable descriptors.
- Run `agentnet inspect` and `agentnet doctor` in release automation.
