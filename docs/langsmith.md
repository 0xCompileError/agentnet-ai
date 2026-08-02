# LangSmith Guide

AgentNet records normalized traces without requiring LangSmith at import time.
Use `InMemoryTracer`, `RunContext`, and `trace_from_context` to collect spans and
events, then pass the trace to `LangSmithExporter` with an injected client.

## Collect A Trace

```python
import agentnet as an

context = an.RunContext(run_id="run-001")
tracer = an.InMemoryTracer()
agent = an.ReActAgent(
    "planner",
    llms=[an.FakeLLM(["ok"], name="strong")],
)

an.run(agent, "input", context=context, tracer=tracer)
trace = an.trace_from_context(context)

print(trace.metrics.to_dict())
```

Runtime traces can include module spans, LLM events, retry events, scheduler
events, tool events, MCP events, topology events, token counts, cost, and
latency. Prompt and completion content are not required for trace metrics.

## Export To LangSmith

`LangSmithExporter` adapts AgentNet traces to an injected client. The client must
offer a `create_run` method compatible with your LangSmith SDK wrapper.

```python
import agentnet as an


class LangSmithClient:
    def create_run(self, **payload):
        print(payload["name"], payload["run_type"])


exporter = an.LangSmithExporter(
    client=LangSmithClient(),
    project_name="agentnet-production",
)

exported_runs = exporter.export(trace)
print(len(exported_runs))
```

The injected client owns credentials, workspace selection, retries, and network
configuration. Do not put those values in AgentNet artifacts.

## Exporter Boundary

The exporter maps AgentNet spans to LangSmith run payloads. Use attributes for
stable identifiers such as module name, module type, scheduler, or environment:

```python
import agentnet as an

span = an.TraceSpan.start(
    "planner",
    run_id="run-001",
    kind="agent",
    attributes={"agentnet.agent": "planner"},
)
span.finish(status="ok")

trace = an.Trace(run_id="run-001", spans=[span])
an.LangSmithExporter(client=LangSmithClient()).export(trace)
```

Use `OpenTelemetryExporter` when the destination is an OTLP pipeline rather than
LangSmith.
