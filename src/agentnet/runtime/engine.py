"""Runtime execution entrypoints."""

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from agentnet.constraints import Constraint, validate_runtime_constraints
from agentnet.core import Module, RunContext
from agentnet.policies import RetryPolicy
from agentnet.tracing import InMemoryTracer


def _ensure_context(context: RunContext | None) -> RunContext:
    return context if context is not None else RunContext(run_id=str(uuid4()))


async def arun(
    module: Module,
    input: Any,
    context: RunContext | None = None,
    *,
    constraints: Iterable[Constraint] | None = None,
    scheduler: Any | None = None,
    retry_policy: RetryPolicy | None = None,
    tracer: InMemoryTracer | None = None,
) -> Any:
    run_context = _ensure_context(context)
    run_context.raise_if_cancelled()
    span = _start_runtime_span(tracer, module, run_context)
    try:
        validate_runtime_constraints(module, constraints, run_context)
        if scheduler is not None:
            result = await scheduler.arun(
                module,
                input,
                run_context,
                retry_policy=retry_policy,
            )
            output = result.output
        else:
            output = await module.arun(input, run_context)
    except Exception as exc:
        _end_runtime_span(tracer, span, run_context, status="error", error=exc)
        raise
    _end_runtime_span(tracer, span, run_context, status="ok")
    return output


def run(
    module: Module,
    input: Any,
    context: RunContext | None = None,
    *,
    constraints: Iterable[Constraint] | None = None,
    scheduler: Any | None = None,
    retry_policy: RetryPolicy | None = None,
    tracer: InMemoryTracer | None = None,
) -> Any:
    run_context = _ensure_context(context)
    run_context.raise_if_cancelled()
    span = _start_runtime_span(tracer, module, run_context)
    try:
        validate_runtime_constraints(module, constraints, run_context)
        if scheduler is not None:
            result = scheduler.run(
                module,
                input,
                run_context,
                retry_policy=retry_policy,
            )
            output = result.output
        else:
            output = module.run(input, run_context)
    except Exception as exc:
        _end_runtime_span(tracer, span, run_context, status="error", error=exc)
        raise
    _end_runtime_span(tracer, span, run_context, status="ok")
    return output


def _start_runtime_span(
    tracer: InMemoryTracer | None,
    module: Module,
    context: RunContext,
) -> Any | None:
    if tracer is None:
        return None
    return tracer.start_span(
        module.name,
        kind="module",
        context=context,
        attributes={
            "agentnet.module": module.name,
            "agentnet.module_type": module.__class__.__name__,
        },
    )


def _end_runtime_span(
    tracer: InMemoryTracer | None,
    span: Any | None,
    context: RunContext,
    *,
    status: str,
    error: Exception | None = None,
) -> None:
    if tracer is None or span is None:
        return
    tracer.end_span(span, status=status, error=error, context=context)
