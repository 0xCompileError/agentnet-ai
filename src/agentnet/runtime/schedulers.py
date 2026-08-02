"""Scheduler abstractions for local and distributed node execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import anyio

from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetExecutionError,
    Module,
    RunContext,
)
from agentnet.mcp._security import validate_safe_metadata
from agentnet.policies import RetryPolicy


@dataclass(frozen=True, slots=True, init=False)
class NodeSpec:
    """Executable node plus descriptor-safe scheduling metadata."""

    module: Module
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        module: Module,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(module, Module):
            raise AgentNetConfigurationError("NodeSpec module must be a Module")
        node_name = name or module.name
        if not node_name:
            raise AgentNetConfigurationError("NodeSpec name cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="NodeSpec")
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "name", node_name)
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.copy(),
            "module": {
                "name": self.module.name,
                "type": self.module.__class__.__name__,
            },
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class NodeFuture:
    """Scheduler future handle for one submitted node."""

    node: NodeSpec
    scheduler: str
    context: RunContext
    handle: Any = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "run_id": self.context.run_id,
            "scheduler": self.scheduler,
        }


@dataclass(frozen=True, slots=True)
class NodeResult:
    """Result of one scheduled node execution."""

    node_name: str
    output: Any
    scheduler: str
    succeeded: bool = True
    attempts: int = 1
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="NodeResult")
        object.__setattr__(self, "attempts", int(self.attempts))
        object.__setattr__(self, "succeeded", bool(self.succeeded))
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "metadata": self.metadata.copy(),
            "node_name": self.node_name,
            "output": self.output,
            "scheduler": self.scheduler,
            "succeeded": self.succeeded,
        }


@runtime_checkable
class Scheduler(Protocol):
    """Protocol implemented by node schedulers."""

    async def submit(
        self,
        node: Module | NodeSpec,
        input: Any,
        context: RunContext | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> NodeFuture:
        """Submit a node for execution."""
        ...

    async def gather(self, futures: Sequence[NodeFuture]) -> list[NodeResult]:
        """Gather submitted node futures."""
        ...


class _BaseScheduler:
    """Shared scheduler implementation for local task-backed schedulers."""

    name: str

    def __init__(
        self,
        *,
        name: str,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("Scheduler name cannot be empty")
        self.name = name
        self.retry_policy = retry_policy

    async def submit(
        self,
        node: Module | NodeSpec,
        input: Any,
        context: RunContext | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> NodeFuture:
        run_context = _ensure_context(context)
        run_context.raise_if_cancelled()
        spec = _coerce_node_spec(node)
        _record_scheduler_event(run_context, self.name, spec.name, "scheduler.submit")
        task = asyncio.create_task(
            self._execute(
                spec,
                input,
                run_context,
                retry_policy=retry_policy or self.retry_policy,
            )
        )
        return NodeFuture(
            node=spec,
            scheduler=self.name,
            context=run_context,
            handle=task,
        )

    async def gather(self, futures: Sequence[NodeFuture]) -> list[NodeResult]:
        for future in futures:
            if future.scheduler != self.name:
                raise AgentNetExecutionError(
                    f"Cannot gather future from scheduler {future.scheduler!r} "
                    f"with scheduler {self.name!r}"
                )
        return list(await asyncio.gather(*(future.handle for future in futures)))

    async def arun(
        self,
        node: Module | NodeSpec,
        input: Any,
        context: RunContext | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> NodeResult:
        future = await self.submit(
            node,
            input,
            context,
            retry_policy=retry_policy,
        )
        return (await self.gather([future]))[0]

    def run(
        self,
        node: Module | NodeSpec,
        input: Any,
        context: RunContext | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> NodeResult:
        async def runner() -> NodeResult:
            return await self.arun(
                node,
                input,
                context,
                retry_policy=retry_policy,
            )

        return anyio.run(runner)

    async def _execute(
        self,
        spec: NodeSpec,
        input: Any,
        context: RunContext,
        *,
        retry_policy: RetryPolicy | None,
    ) -> NodeResult:
        attempts = _transport_attempt_limit(retry_policy)
        for attempt in range(1, attempts + 1):
            context.raise_if_cancelled()
            _record_scheduler_event(
                context,
                self.name,
                spec.name,
                "scheduler.started",
                attempt=attempt,
            )
            try:
                output = await self._run_once(spec, input, context)
            except Exception as exc:
                if attempt >= attempts:
                    _record_scheduler_event(
                        context,
                        self.name,
                        spec.name,
                        "scheduler.failed",
                        attempt=attempt,
                        error_type=type(exc).__name__,
                    )
                    raise
                delay_seconds = _retry_delay(retry_policy, attempt)
                _record_scheduler_retry_event(
                    context,
                    self.name,
                    spec.name,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    delay_seconds=delay_seconds,
                    error_type=type(exc).__name__,
                )
                if delay_seconds:
                    await anyio.sleep(delay_seconds)
                continue

            _record_scheduler_event(
                context,
                self.name,
                spec.name,
                "scheduler.completed",
                attempt=attempt,
            )
            return NodeResult(
                node_name=spec.name,
                output=output,
                scheduler=self.name,
                attempts=attempt,
                metadata=spec.metadata,
            )

        raise AgentNetExecutionError("Scheduler attempt budget was exhausted")

    async def _run_once(
        self,
        spec: NodeSpec,
        input: Any,
        context: RunContext,
    ) -> Any:
        raise NotImplementedError


class LocalScheduler(_BaseScheduler):
    """Execute nodes in the current event loop."""

    def __init__(self, *, retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(name="local", retry_policy=retry_policy)

    async def _run_once(
        self,
        spec: NodeSpec,
        input: Any,
        context: RunContext,
    ) -> Any:
        return await spec.module.arun(input, context)


class ThreadPoolScheduler(_BaseScheduler):
    """Execute nodes in a standard-library thread pool."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(name="thread_pool", retry_policy=retry_policy)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agentnet-thread",
        )

    async def _run_once(
        self,
        spec: NodeSpec,
        input: Any,
        context: RunContext,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            _run_module_sync,
            spec.module,
            input,
            context,
        )

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


class ProcessPoolScheduler(_BaseScheduler):
    """Execute picklable nodes in a standard-library process pool."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(name="process_pool", retry_policy=retry_policy)
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    async def _run_once(
        self,
        spec: NodeSpec,
        input: Any,
        context: RunContext,
    ) -> Any:
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            self._executor,
            _run_module_in_process,
            spec.module,
            input,
            context.to_dict(),
        )
        _merge_context(context, RunContext.from_dict(payload["context"]))
        return payload["output"]

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


class _InjectedClientScheduler(_BaseScheduler):
    """Base class for optional remote scheduler integrations."""

    def __init__(
        self,
        *,
        client: Any,
        name: str,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if client is None:
            raise AgentNetConfigurationError(
                f"{self.__class__.__name__} requires an injected client"
            )
        if not hasattr(client, "submit"):
            raise AgentNetConfigurationError(
                f"{self.__class__.__name__} client must implement submit"
            )
        super().__init__(name=name, retry_policy=retry_policy)
        self.client = client

    async def _run_once(
        self,
        spec: NodeSpec,
        input: Any,
        context: RunContext,
    ) -> Any:
        handle = await _maybe_await(
            self.client.submit(spec.module, input, context.to_dict())
        )
        payload = handle
        if hasattr(self.client, "gather"):
            gathered = await _maybe_await(self.client.gather([handle]))
            payload = list(gathered)[0]
        return _coerce_remote_output(payload, spec, context, self.name)


class RayScheduler(_InjectedClientScheduler):
    """Adapter for Ray-style clients supplied by application code."""

    def __init__(self, *, client: Any, retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(client=client, name="ray", retry_policy=retry_policy)


class CeleryScheduler(_InjectedClientScheduler):
    """Adapter for Celery-style clients supplied by application code."""

    def __init__(self, *, client: Any, retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(client=client, name="celery", retry_policy=retry_policy)


class TemporalScheduler(_InjectedClientScheduler):
    """Adapter for Temporal-style clients supplied by application code."""

    def __init__(self, *, client: Any, retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(client=client, name="temporal", retry_policy=retry_policy)


def _ensure_context(context: RunContext | None) -> RunContext:
    return context if context is not None else RunContext(run_id=str(uuid4()))


def _coerce_node_spec(node: Module | NodeSpec) -> NodeSpec:
    if isinstance(node, NodeSpec):
        return node
    if isinstance(node, Module):
        return NodeSpec(node)
    raise AgentNetConfigurationError("Scheduler node must be a Module or NodeSpec")


def _transport_attempt_limit(retry_policy: RetryPolicy | None) -> int:
    if retry_policy is None:
        return 1
    attempts = 1 + int(retry_policy.transport_retries)
    if retry_policy.max_total_attempts is not None:
        attempts = min(attempts, retry_policy.max_total_attempts)
    return max(1, attempts)


def _retry_delay(retry_policy: RetryPolicy | None, retry_number: int) -> float:
    if retry_policy is None:
        return 0.0
    return float(retry_policy.backoff_delay(retry_number))


def _record_scheduler_event(
    context: RunContext,
    scheduler: str,
    node_name: str,
    event_type: str,
    *,
    attempt: int | None = None,
    error_type: str | None = None,
) -> None:
    events = context.metadata.setdefault("scheduler_events", [])
    if not isinstance(events, list):
        events = []
        context.metadata["scheduler_events"] = events
    event: dict[str, Any] = {
        "node": node_name,
        "run_id": context.run_id,
        "scheduler": scheduler,
        "type": event_type,
    }
    if attempt is not None:
        event["attempt"] = attempt
    if error_type is not None:
        event["error_type"] = error_type
    events.append(event)


def _record_scheduler_retry_event(
    context: RunContext,
    scheduler: str,
    node_name: str,
    *,
    attempt: int,
    next_attempt: int,
    delay_seconds: float,
    error_type: str,
) -> None:
    events = context.metadata.setdefault("scheduler_retry_events", [])
    if not isinstance(events, list):
        events = []
        context.metadata["scheduler_retry_events"] = events
    events.append(
        {
            "attempt": attempt,
            "delay_seconds": delay_seconds,
            "error_type": error_type,
            "next_attempt": next_attempt,
            "node": node_name,
            "reason": "transport",
            "run_id": context.run_id,
            "scheduler": scheduler,
            "type": "scheduler.retry.started",
        }
    )
    metrics = context.metadata.setdefault(
        "scheduler_metrics",
        {
            "total_backoff_seconds": 0.0,
            "total_retries": 0,
            "transport_retries": 0,
        },
    )
    if isinstance(metrics, dict):
        metrics["total_backoff_seconds"] = (
            float(metrics.get("total_backoff_seconds", 0.0)) + delay_seconds
        )
        metrics["total_retries"] = int(metrics.get("total_retries", 0)) + 1
        metrics["transport_retries"] = int(metrics.get("transport_retries", 0)) + 1


def _run_module_sync(module: Module, input: Any, context: RunContext) -> Any:
    return module.run(input, context)


def _run_module_in_process(
    module: Module,
    input: Any,
    context_payload: dict[str, Any],
) -> dict[str, Any]:
    context = RunContext.from_dict(context_payload)
    output = module.run(input, context)
    return {
        "context": context.to_dict(),
        "output": output,
    }


def _merge_context(target: RunContext, source: RunContext) -> None:
    target.graph_state = source.graph_state
    target.cancelled = source.cancelled
    target.metadata.update(source.metadata)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _coerce_remote_output(
    payload: Any,
    spec: NodeSpec,
    context: RunContext,
    scheduler: str,
) -> Any:
    if isinstance(payload, NodeResult):
        return payload.output
    if isinstance(payload, Mapping):
        remote_context = payload.get("context")
        if isinstance(remote_context, dict):
            _merge_context(context, RunContext.from_dict(remote_context))
        if payload.get("succeeded", True) is False:
            error_type = str(payload.get("error_type", "RemoteExecutionError"))
            message = str(payload.get("error_message", "Remote scheduler failed"))
            raise AgentNetExecutionError(f"{scheduler} node {spec.name!r} failed: {message}") from (
                RuntimeError(error_type)
            )
        return payload.get("output")
    return payload
