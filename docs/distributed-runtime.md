# Distributed Runtime Guide

The distributed runtime centers on the `Scheduler` protocol. A scheduler accepts
a `Module` or `NodeSpec`, runs it, and returns `NodeResult` records. Local
schedulers ship with AgentNet; remote integrations use injected clients.

## Local Schedulers

`LocalScheduler` runs in the current event loop. `ThreadPoolScheduler` uses the
standard-library thread pool. `ProcessPoolScheduler` uses a process pool and
requires picklable modules and inputs.

```python
import agentnet as an

agent = an.ReActAgent(
    "planner",
    llms=[an.FakeLLM(["done"], name="strong")],
)

scheduler = an.ThreadPoolScheduler(max_workers=2)
try:
    output = an.run(agent, "plan this", scheduler=scheduler)
finally:
    scheduler.shutdown()

print(output)
```

Pass `RetryPolicy` to retry scheduler-level transport failures:

```python
import agentnet as an

policy = an.RetryPolicy(transport_retries=2, backoff_base_seconds=0.0)
output = an.run(agent, "plan this", scheduler=an.LocalScheduler(), retry_policy=policy)
```

## Remote Scheduler Adapters

`RayScheduler`, `CeleryScheduler`, and `TemporalScheduler` are dependency-light
adapters. They do not import or configure those frameworks. They require
injected clients that implement `submit`, and optionally `gather`.

```python
import agentnet as an


class RemoteClient:
    def submit(self, module, input, context):
        return {"output": module.run(input, an.RunContext.from_dict(context))}


scheduler = an.RayScheduler(client=RemoteClient())
result = an.run(agent, "remote plan", scheduler=scheduler)
```

Use injected clients to keep framework setup, credentials, queues, namespaces,
and deployment-specific routing outside AgentNet descriptors.

## Node Metadata

`NodeSpec` carries descriptor-safe scheduling metadata:

```python
import agentnet as an

spec = an.NodeSpec(agent, name="planner", metadata={"queue": "cpu"})
future_result = an.LocalScheduler().run(spec, "input")

print(future_result.to_dict())
```

Metadata is validated so scheduler descriptors do not carry secret-like keys.

## Tracing

Schedulers record submit, start, completion, failure, and retry metadata on
`RunContext`. Convert that metadata into a normalized trace:

```python
import agentnet as an

context = an.RunContext(run_id="distributed-run")
an.run(agent, "input", context=context, scheduler=an.LocalScheduler())
trace = an.trace_from_context(context)

print(trace.metrics.scheduler_retry_count)
```
