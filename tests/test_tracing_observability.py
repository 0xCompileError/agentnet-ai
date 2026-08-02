from __future__ import annotations

from typing import Any

import pytest

import agentnet as an


class UsageLLM:
    def __init__(self) -> None:
        self.name = "strong"
        self.model = "fake-model"

    async def complete(self, request: an.ChatRequest) -> an.ChatResponse:
        return an.ChatResponse(
            content="ok",
            model=request.model,
            usage={
                "completion_tokens": 3,
                "prompt_tokens": 7,
                "total_tokens": 10,
            },
            metadata={"cost_usd": 0.015},
        )


class EchoModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return input


class FakeLangSmithClient:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def create_run(self, **payload: Any) -> None:
        self.runs.append(payload)


class FakeOTelExporter:
    def __init__(self) -> None:
        self.exported: list[dict[str, Any]] = []

    def export(self, spans: list[dict[str, Any]]) -> None:
        self.exported.extend(spans)


def test_trace_event_and_span_round_trip_without_secret_attributes() -> None:
    event = an.TraceEvent(
        "llm.completed",
        run_id="run-1",
        attributes={
            "cost_usd": 0.01,
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens": 4,
                "total_tokens": 6,
            },
        },
    )
    span = an.TraceSpan.start(
        "planner",
        run_id="run-1",
        kind="agent",
        attributes={"agent": "planner"},
    )
    span.finish(status="ok")

    assert an.TraceEvent.from_dict(event.to_dict()).to_dict() == event.to_dict()
    assert an.TraceSpan.from_dict(span.to_dict()).to_dict() == span.to_dict()
    assert span.duration_ms is not None

    with pytest.raises(an.AgentNetConfigurationError, match="may serialize secrets"):
        an.TraceEvent("unsafe", attributes={"api_token": "secret"})


def test_runtime_tracer_records_span_and_llm_cost_token_latency_metrics() -> None:
    context = an.RunContext(run_id="run-llm")
    tracer = an.InMemoryTracer()
    agent = an.ReActAgent("planner", llms=[UsageLLM()])

    output = an.run(agent, "input", context=context, tracer=tracer)
    trace = an.trace_from_context(context)

    assert output == "ok"
    assert trace.run_id == "run-llm"
    assert trace.spans[0].name == "planner"
    assert trace.metrics.prompt_tokens == 7
    assert trace.metrics.completion_tokens == 3
    assert trace.metrics.total_tokens == 10
    assert trace.metrics.total_cost_usd == 0.015
    assert trace.metrics.total_latency_ms > 0
    assert [event.event_type for event in trace.events if event.event_type == "llm.completed"]


def test_trace_from_context_normalizes_existing_runtime_metadata() -> None:
    context = an.RunContext(
        run_id="run-existing",
        metadata={
            "mcp_events": [
                {
                    "agent": "operator",
                    "approved": True,
                    "descriptor_hash": "sha256:abc",
                    "server": "github",
                    "tool": "search",
                    "type": "mcp.tool.called",
                }
            ],
            "retry_events": [
                {
                    "agent": "planner",
                    "attempt": 1,
                    "delay_seconds": 0.0,
                    "error_type": "TimeoutError",
                    "model": "strong",
                    "next_attempt": 2,
                    "reason": "transport",
                    "type": "retry.started",
                }
            ],
            "scheduler_retry_events": [
                {
                    "attempt": 1,
                    "delay_seconds": 0.0,
                    "error_type": "TimeoutError",
                    "next_attempt": 2,
                    "node": "planner",
                    "reason": "transport",
                    "run_id": "run-existing",
                    "scheduler": "local",
                    "type": "scheduler.retry.started",
                }
            ],
            "tool_events": [
                {
                    "agent": "operator",
                    "side_effect": True,
                    "tool": "create_ticket",
                    "type": "tool.called",
                }
            ],
        },
    )

    trace = an.trace_from_context(context)

    assert {event.event_type for event in trace.events} >= {
        "mcp.tool.called",
        "retry.started",
        "scheduler.retry.started",
        "tool.called",
    }
    assert trace.metrics.retry_count == 1
    assert trace.metrics.scheduler_retry_count == 1
    assert trace.metrics.tool_call_count == 1
    assert trace.metrics.mcp_tool_call_count == 1


def test_langsmith_exporter_uses_injected_client_without_importing_langsmith() -> None:
    client = FakeLangSmithClient()
    span = an.TraceSpan.start("planner", run_id="run-export", kind="agent")
    span.finish(status="ok")
    trace = an.Trace(run_id="run-export", spans=[span])

    exported = an.LangSmithExporter(client=client, project_name="agentnet-test").export(trace)

    assert exported[0]["name"] == "planner"
    assert exported[0]["run_type"] == "agent"
    assert exported[0]["project_name"] == "agentnet-test"
    assert client.runs == exported


def test_opentelemetry_exporter_uses_injected_exporter_without_dependency() -> None:
    exporter = FakeOTelExporter()
    span = an.TraceSpan.start(
        "planner",
        run_id="run-otel",
        kind="agent",
        attributes={"agentnet.agent": "planner"},
    )
    span.finish(status="ok")
    trace = an.Trace(run_id="run-otel", spans=[span])

    exported = an.OpenTelemetryExporter(exporter=exporter).export(trace)

    assert exported[0]["name"] == "planner"
    assert exported[0]["attributes"]["agentnet.run_id"] == "run-otel"
    assert exported[0]["attributes"]["agentnet.agent"] == "planner"
    assert exporter.exported == exported


def test_topology_tracking_records_trials_and_best_score() -> None:
    seed = EchoModule("seed")
    branch = EchoModule("branch")
    optimizer = an.TopologyOptimizer(
        search_space=an.TopologySearchSpace(
            branch_candidates=[branch],
            max_trials=2,
        )
    )
    result = optimizer.search(
        seed,
        scorer=an.ArchitectureScorer(base_scorer=lambda graph: float(len(graph.nodes))),
    )
    context = an.RunContext(run_id="run-topology")

    an.record_topology_result(context, result)
    trace = an.trace_from_context(context)

    assert trace.metrics.topology_trial_count == len(result.checkpoints)
    assert trace.metrics.topology_best_score == result.score
    assert "topology.best" in {event.event_type for event in trace.events}


def test_tracing_public_exports_are_available() -> None:
    exported: list[Any] = [
        an.InMemoryTracer,
        an.LangSmithExporter,
        an.OpenTelemetryExporter,
        an.Trace,
        an.TraceEvent,
        an.TraceMetrics,
        an.TraceSpan,
        an.record_topology_result,
        an.trace_from_context,
    ]

    assert all(value is not None for value in exported)
