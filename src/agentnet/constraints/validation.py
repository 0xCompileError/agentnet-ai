"""Constraint validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

from agentnet.constraints.base import Constraint, ConstraintResult
from agentnet.core import AgentNetConfigurationError, AgentNetValidationError, RunContext


def validate_runtime_constraints(
    candidate: Any,
    constraints: Iterable[Constraint] | None,
    context: RunContext | None = None,
) -> tuple[ConstraintResult, ...]:
    """Evaluate runtime constraints and reject blocking hard failures."""

    results = _evaluate_constraints(candidate, constraints, context)
    if context is not None:
        _record_constraint_summaries(
            context.metadata,
            "constraint_results",
            results,
        )

    blocking = tuple(result for result in results if result.blocks_candidate)
    if blocking:
        names = ", ".join(result.constraint for result in blocking)
        raise AgentNetValidationError(
            f"Runtime constraint validation failed: {names}"
        )
    return results


def validate_training_constraints(
    candidate: Any,
    constraints: Iterable[Constraint] | None,
    *,
    metadata: MutableMapping[str, Any] | None = None,
) -> tuple[ConstraintResult, ...]:
    """Evaluate training-time constraints and reject blocking hard failures."""

    results = _evaluate_constraints(candidate, constraints, None)
    if metadata is not None:
        _record_constraint_summaries(
            metadata,
            "training_constraint_results",
            results,
        )

    blocking = tuple(result for result in results if result.blocks_candidate)
    if blocking:
        names = ", ".join(result.constraint for result in blocking)
        raise AgentNetValidationError(
            f"Training constraint validation failed: {names}"
        )
    return results


def _evaluate_constraints(
    candidate: Any,
    constraints: Iterable[Constraint] | None,
    context: RunContext | None,
) -> tuple[ConstraintResult, ...]:
    constraint_list = tuple(constraints or ())
    for constraint in constraint_list:
        if not isinstance(constraint, Constraint):
            raise AgentNetConfigurationError(
                "Constraints must be Constraint instances"
            )

    return tuple(
        constraint.evaluate(candidate, context) for constraint in constraint_list
    )


def _record_constraint_summaries(
    metadata: MutableMapping[str, Any],
    key: str,
    results: tuple[ConstraintResult, ...],
) -> None:
    summaries = metadata.setdefault(key, [])
    summaries.extend(_constraint_summary(result) for result in results)


def _constraint_summary(result: ConstraintResult) -> dict[str, Any]:
    return {
        "blocks_candidate": result.blocks_candidate,
        "constraint": result.constraint,
        "kind": result.kind.value,
        "message": result.message,
        "passed": result.passed,
    }
