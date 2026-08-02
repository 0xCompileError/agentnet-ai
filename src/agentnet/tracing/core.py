"""Dependency-light tracing and observability primitives."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Self
from uuid import uuid4

from agentnet.core import AgentNetConfigurationError, RunContext
from agentnet.mcp._security import validate_descriptor_payload_no_secrets

_LEGACY_EVENT_KEYS = (
    "llm_events",
    "mcp_events",
    "retry_events",
    "scheduler_events",
    "scheduler_retry_events",
    "tool_events",
    "topology_events",
)
_SECRET_KEY_PARTS = ("api_key", "password", "secret", "token")
_SAFE_TOKEN_METRIC_KEYS = {
    "cached_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "total_tokens",
}


@dataclass(slots=True)
class TraceEvent:
    """A descriptor-safe tracing event."""

    event_type: str
    run_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    timestamp_unix_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.event_type:
            raise AgentNetConfigurationError("TraceEvent event_type cannot be empty")
        timestamp_ns = (
            self.timestamp_unix_ns
            if self.timestamp_unix_ns is not None
            else time.time_ns()
        )
        timestamp = self.timestamp or _iso_from_ns(timestamp_ns)
        attributes = dict(self.attributes)
        _validate_trace_payload_no_secrets(attributes, label="TraceEvent")
        self.attributes = attributes
        self.timestamp = timestamp
        self.timestamp_unix_ns = int(timestamp_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": dict(self.attributes),
            "parent_span_id": self.parent_span_id,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "timestamp": self.timestamp,
            "timestamp_unix_ns": self.timestamp_unix_ns,
            "type": self.event_type,
        }

    @classmethod
    def from_dict(cls, event: Mapping[str, Any]) -> Self:
        return cls(
            str(event["type"]),
            run_id=None if event.get("run_id") is None else str(event["run_id"]),
            span_id=None if event.get("span_id") is None else str(event["span_id"]),
            parent_span_id=(
                None
                if event.get("parent_span_id") is None
                else str(event["parent_span_id"])
            ),
            attributes=dict(event.get("attributes", {})),
            timestamp=None if event.get("timestamp") is None else str(event["timestamp"]),
            timestamp_unix_ns=(
                None
                if event.get("timestamp_unix_ns") is None
                else int(event["timestamp_unix_ns"])
            ),
        )


@dataclass(slots=True)
class TraceSpan:
    """A descriptor-safe tracing span."""

    span_id: str
    run_id: str
    name: str
    kind: str = "internal"
    parent_span_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    started_at_unix_ns: int | None = None
    ended_at: str | None = None
    ended_at_unix_ns: int | None = None
    status: str = "running"

    def __post_init__(self) -> None:
        if not self.span_id:
            raise AgentNetConfigurationError("TraceSpan span_id cannot be empty")
        if not self.run_id:
            raise AgentNetConfigurationError("TraceSpan run_id cannot be empty")
        if not self.name:
            raise AgentNetConfigurationError("TraceSpan name cannot be empty")
        if not self.kind:
            raise AgentNetConfigurationError("TraceSpan kind cannot be empty")
        start_ns = (
            self.started_at_unix_ns
            if self.started_at_unix_ns is not None
            else time.time_ns()
        )
        attributes = dict(self.attributes)
        _validate_trace_payload_no_secrets(attributes, label="TraceSpan")
        self.attributes = attributes
        self.started_at_unix_ns = int(start_ns)
        self.started_at = self.started_at or _iso_from_ns(start_ns)
        if self.ended_at_unix_ns is not None and self.ended_at is None:
            self.ended_at = _iso_from_ns(self.ended_at_unix_ns)

    @classmethod
    def start(
        cls,
        name: str,
        *,
        run_id: str,
        kind: str = "internal",
        parent_span_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Self:
        return cls(
            span_id=str(uuid4()),
            run_id=run_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=dict(attributes or {}),
        )

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at_unix_ns is None or self.started_at_unix_ns is None:
            return None
        return max(0.0, (self.ended_at_unix_ns - self.started_at_unix_ns) / 1_000_000)

    def finish(
        self,
        *,
        status: str = "ok",
        error: Exception | None = None,
    ) -> None:
        end_ns = max(time.time_ns(), int(self.started_at_unix_ns or 0) + 1)
        attributes = dict(self.attributes)
        if error is not None:
            attributes["error_type"] = type(error).__name__
        _validate_trace_payload_no_secrets(attributes, label="TraceSpan")
        self.attributes = attributes
        self.status = status
        self.ended_at_unix_ns = end_ns
        self.ended_at = _iso_from_ns(end_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": dict(self.attributes),
            "ended_at": self.ended_at,
            "ended_at_unix_ns": self.ended_at_unix_ns,
            "kind": self.kind,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "started_at": self.started_at,
            "started_at_unix_ns": self.started_at_unix_ns,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, span: Mapping[str, Any]) -> Self:
        return cls(
            span_id=str(span["span_id"]),
            run_id=str(span["run_id"]),
            name=str(span["name"]),
            kind=str(span.get("kind", "internal")),
            parent_span_id=(
                None if span.get("parent_span_id") is None else str(span["parent_span_id"])
            ),
            attributes=dict(span.get("attributes", {})),
            started_at=None if span.get("started_at") is None else str(span["started_at"]),
            started_at_unix_ns=(
                None
                if span.get("started_at_unix_ns") is None
                else int(span["started_at_unix_ns"])
            ),
            ended_at=None if span.get("ended_at") is None else str(span["ended_at"]),
            ended_at_unix_ns=(
                None
                if span.get("ended_at_unix_ns") is None
                else int(span["ended_at_unix_ns"])
            ),
            status=str(span.get("status", "running")),
        )


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    """Aggregated trace metrics."""

    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    retry_count: int = 0
    scheduler_retry_count: int = 0
    tool_call_count: int = 0
    mcp_tool_call_count: int = 0
    topology_trial_count: int = 0
    topology_best_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion_tokens": self.completion_tokens,
            "mcp_tool_call_count": self.mcp_tool_call_count,
            "prompt_tokens": self.prompt_tokens,
            "retry_count": self.retry_count,
            "scheduler_retry_count": self.scheduler_retry_count,
            "tool_call_count": self.tool_call_count,
            "topology_best_score": self.topology_best_score,
            "topology_trial_count": self.topology_trial_count,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True, init=False)
class Trace:
    """A normalized view of one AgentNet run."""

    run_id: str
    spans: tuple[TraceSpan, ...]
    events: tuple[TraceEvent, ...]
    metrics: TraceMetrics

    def __init__(
        self,
        *,
        run_id: str,
        spans: Sequence[TraceSpan] | None = None,
        events: Sequence[TraceEvent] | None = None,
        metrics: TraceMetrics | None = None,
    ) -> None:
        span_tuple = tuple(spans or ())
        event_tuple = tuple(events or ())
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "spans", span_tuple)
        object.__setattr__(self, "events", event_tuple)
        object.__setattr__(
            self,
            "metrics",
            metrics or _compute_metrics(span_tuple, event_tuple),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "metrics": self.metrics.to_dict(),
            "run_id": self.run_id,
            "spans": [span.to_dict() for span in self.spans],
        }


class InMemoryTracer:
    """Simple tracer that records spans and events into memory and run metadata."""

    def __init__(self, *, run_id: str | None = None) -> None:
        self.run_id = run_id
        self._spans: list[TraceSpan] = []
        self._events: list[TraceEvent] = []

    @property
    def spans(self) -> tuple[TraceSpan, ...]:
        return tuple(self._spans)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def start_span(
        self,
        name: str,
        *,
        kind: str = "internal",
        context: RunContext | None = None,
        parent_span_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        span = TraceSpan.start(
            name,
            run_id=_resolve_run_id(context, self.run_id),
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        self._spans.append(span)
        _upsert_context_span(context, span)
        self.record_event("span.started", context=context, span=span)
        return span

    def end_span(
        self,
        span: TraceSpan,
        *,
        status: str = "ok",
        error: Exception | None = None,
        context: RunContext | None = None,
    ) -> TraceSpan:
        span.finish(status=status, error=error)
        _upsert_context_span(context, span)
        self.record_event(
            "span.failed" if status == "error" else "span.completed",
            context=context,
            span=span,
        )
        return span

    def record_event(
        self,
        event_type: str,
        *,
        context: RunContext | None = None,
        span: TraceSpan | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_type,
            run_id=_resolve_run_id(context, span.run_id if span is not None else self.run_id),
            span_id=None if span is None else span.span_id,
            parent_span_id=None if span is None else span.parent_span_id,
            attributes=dict(attributes or {}),
        )
        self._events.append(event)
        _append_context_event(context, event)
        return event

    def snapshot(self, context: RunContext | None = None) -> Trace:
        if context is None:
            return Trace(
                run_id=self.run_id or "",
                spans=self._spans,
                events=self._events,
            )
        return trace_from_context(context)


class LangSmithExporter:
    """Exporter that adapts traces to LangSmith-style run payloads."""

    def __init__(self, *, client: Any | None = None, project_name: str | None = None) -> None:
        self.client = client
        self.project_name = project_name

    def export(self, trace: Trace) -> list[dict[str, Any]]:
        payloads = [
            {
                "end_time": span.ended_at,
                "error": span.attributes.get("error_type") if span.status == "error" else None,
                "extra": {
                    "events": [
                        event.to_dict()
                        for event in trace.events
                        if event.span_id == span.span_id
                    ],
                    "metrics": trace.metrics.to_dict(),
                    "metadata": dict(span.attributes),
                },
                "id": span.span_id,
                "name": span.name,
                "parent_run_id": span.parent_span_id,
                "project_name": self.project_name,
                "run_type": span.kind,
                "start_time": span.started_at,
                "tags": ["agentnet"],
            }
            for span in trace.spans
        ]
        _send_to_langsmith_client(self.client, payloads)
        return payloads


class OpenTelemetryExporter:
    """Exporter that adapts traces to OpenTelemetry-style span dictionaries."""

    def __init__(self, *, exporter: Any | None = None, service_name: str = "agentnet") -> None:
        self.exporter = exporter
        self.service_name = service_name

    def export(self, trace: Trace) -> list[dict[str, Any]]:
        spans = [
            {
                "attributes": _otel_attributes(
                    {
                        **dict(span.attributes),
                        "agentnet.kind": span.kind,
                        "agentnet.run_id": span.run_id,
                        "service.name": self.service_name,
                    }
                ),
                "end_time_unix_nano": span.ended_at_unix_ns,
                "events": [
                    {
                        "attributes": _otel_attributes(event.attributes),
                        "name": event.event_type,
                        "time_unix_nano": event.timestamp_unix_ns,
                    }
                    for event in trace.events
                    if event.span_id == span.span_id
                ],
                "name": span.name,
                "parent_span_id": span.parent_span_id,
                "span_id": span.span_id,
                "start_time_unix_nano": span.started_at_unix_ns,
                "status": span.status,
            }
            for span in trace.spans
        ]
        _send_to_otel_exporter(self.exporter, spans)
        return spans


def trace_from_context(context: RunContext) -> Trace:
    """Build a normalized trace from a run context's metadata streams."""

    spans = [
        TraceSpan.from_dict(dict(span))
        for span in _metadata_sequence(context.metadata.get("trace_spans"))
    ]
    events = [
        TraceEvent.from_dict(dict(event))
        for event in _metadata_sequence(context.metadata.get("trace_events"))
    ]
    events.extend(_legacy_events(context))
    return Trace(run_id=context.run_id, spans=spans, events=events)


def record_llm_event(
    context: Any | None,
    *,
    agent_name: str,
    model: str,
    event_type: str,
    attempt: int | None = None,
    cost_usd: float | None = None,
    duration_ms: float | None = None,
    error: Exception | None = None,
    usage: Mapping[str, int] | None = None,
) -> None:
    """Record an LLM observability event on a run context."""

    if context is None or not hasattr(context, "metadata"):
        return
    events = context.metadata.setdefault("llm_events", [])
    if not isinstance(events, list):
        return
    event: dict[str, Any] = {
        "agent": agent_name,
        "model": model,
        "type": event_type,
    }
    if attempt is not None:
        event["attempt"] = attempt
    if cost_usd is not None:
        event["cost_usd"] = float(cost_usd)
    if duration_ms is not None:
        event["duration_ms"] = float(duration_ms)
    if error is not None:
        event["error_type"] = type(error).__name__
    if usage:
        event["usage"] = {str(key): int(value) for key, value in usage.items()}
    _validate_trace_payload_no_secrets(event, label="LLM trace event")
    events.append(event)


def record_topology_result(context: RunContext, result: Any) -> None:
    """Record topology optimization checkpoints and the selected result."""

    events = context.metadata.setdefault("topology_events", [])
    if not isinstance(events, list):
        return
    for checkpoint in getattr(result, "checkpoints", ()):
        mutation = getattr(checkpoint, "mutation", None)
        event = {
            "accepted": bool(getattr(checkpoint, "metadata", {}).get("accepted", True)),
            "mutation": None if mutation is None else mutation.to_dict(),
            "score": float(checkpoint.score.score),
            "trial": int(checkpoint.trial),
            "type": "topology.trial",
        }
        validate_descriptor_payload_no_secrets(event, label="Topology trace event")
        events.append(event)
    best_mutation = getattr(result, "mutation", None)
    best_event = {
        "evaluated_candidates": int(result.metadata.get("evaluated_candidates", 0)),
        "mutation": None if best_mutation is None else best_mutation.to_dict(),
        "rejected_candidates": int(result.metadata.get("rejected_candidates", 0)),
        "score": float(result.score),
        "type": "topology.best",
    }
    validate_descriptor_payload_no_secrets(best_event, label="Topology trace event")
    events.append(best_event)


def _compute_metrics(
    spans: Sequence[TraceSpan],
    events: Sequence[TraceEvent],
) -> TraceMetrics:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    total_latency_ms = sum(span.duration_ms or 0.0 for span in spans)
    retry_count = 0
    scheduler_retry_count = 0
    tool_call_count = 0
    mcp_tool_call_count = 0
    topology_trial_count = 0
    topology_best_score: float | None = None

    for event in events:
        attributes = event.attributes
        total_cost_usd += _event_cost(attributes)
        usage = attributes.get("usage")
        if isinstance(usage, Mapping):
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            total_tokens += int(
                usage.get(
                    "total_tokens",
                    int(usage.get("prompt_tokens", 0))
                    + int(usage.get("completion_tokens", 0)),
                )
            )
        prompt_tokens += int(attributes.get("prompt_tokens", 0))
        completion_tokens += int(attributes.get("completion_tokens", 0))
        total_tokens += int(attributes.get("total_tokens", 0))
        total_latency_ms += float(attributes.get("duration_ms", 0.0))

        if event.event_type.startswith("retry."):
            retry_count += 1
        if event.event_type == "scheduler.retry.started":
            scheduler_retry_count += 1
        if event.event_type == "tool.called":
            tool_call_count += 1
        if event.event_type == "mcp.tool.called":
            mcp_tool_call_count += 1
        if event.event_type == "topology.trial":
            topology_trial_count += 1
        if event.event_type == "topology.best":
            score = attributes.get("score")
            if score is not None:
                topology_best_score = float(score)

    return TraceMetrics(
        completion_tokens=completion_tokens,
        mcp_tool_call_count=mcp_tool_call_count,
        prompt_tokens=prompt_tokens,
        retry_count=retry_count,
        scheduler_retry_count=scheduler_retry_count,
        tool_call_count=tool_call_count,
        topology_best_score=topology_best_score,
        topology_trial_count=topology_trial_count,
        total_cost_usd=total_cost_usd,
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
    )


def _event_cost(attributes: Mapping[str, Any]) -> float:
    for key in ("cost_usd", "total_cost_usd", "cost"):
        value = attributes.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _validate_trace_payload_no_secrets(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized == "env":
                raise AgentNetConfigurationError(
                    f"{label} cannot include env because it may serialize secrets"
                )
            if (
                normalized not in _SAFE_TOKEN_METRIC_KEYS
                and any(part in normalized for part in _SECRET_KEY_PARTS)
            ):
                raise AgentNetConfigurationError(
                    f"{label} metadata key {key!r} may serialize secrets"
                )
            _validate_trace_payload_no_secrets(nested, label=label)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for nested in value:
            _validate_trace_payload_no_secrets(nested, label=label)


def _legacy_events(context: RunContext) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for key in _LEGACY_EVENT_KEYS:
        for event in _metadata_sequence(context.metadata.get(key)):
            event_payload = dict(event)
            event_type = str(event_payload.pop("type"))
            run_id = str(event_payload.pop("run_id", context.run_id))
            span_id = event_payload.pop("span_id", None)
            parent_span_id = event_payload.pop("parent_span_id", None)
            events.append(
                TraceEvent(
                    event_type,
                    run_id=run_id,
                    span_id=None if span_id is None else str(span_id),
                    parent_span_id=(
                        None if parent_span_id is None else str(parent_span_id)
                    ),
                    attributes=event_payload,
                )
            )
    return events


def _metadata_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _resolve_run_id(context: RunContext | None, fallback: str | None) -> str:
    if context is not None:
        return context.run_id
    if fallback:
        return fallback
    return str(uuid4())


def _append_context_event(context: RunContext | None, event: TraceEvent) -> None:
    if context is None:
        return
    events = context.metadata.setdefault("trace_events", [])
    if isinstance(events, list):
        events.append(event.to_dict())


def _upsert_context_span(context: RunContext | None, span: TraceSpan) -> None:
    if context is None:
        return
    spans = context.metadata.setdefault("trace_spans", [])
    if not isinstance(spans, list):
        return
    replacement = span.to_dict()
    for index, existing in enumerate(spans):
        if isinstance(existing, Mapping) and existing.get("span_id") == span.span_id:
            spans[index] = replacement
            return
    spans.append(replacement)


def _send_to_langsmith_client(client: Any | None, payloads: list[dict[str, Any]]) -> None:
    if client is None:
        return
    if hasattr(client, "create_run"):
        for payload in payloads:
            client.create_run(**payload)
        return
    if hasattr(client, "batch_ingest"):
        client.batch_ingest(payloads)
        return
    if callable(client):
        client(payloads)
        return
    raise AgentNetConfigurationError("LangSmithExporter client is not callable")


def _send_to_otel_exporter(exporter: Any | None, spans: list[dict[str, Any]]) -> None:
    if exporter is None:
        return
    if hasattr(exporter, "export"):
        exporter.export(spans)
        return
    if callable(exporter):
        exporter(spans)
        return
    raise AgentNetConfigurationError("OpenTelemetryExporter exporter is not callable")


def _otel_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    otel: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, str | int | float | bool) or value is None:
            otel[str(key)] = value
        else:
            otel[str(key)] = json.dumps(value, sort_keys=True, default=str)
    return otel


def _iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, UTC).isoformat()
