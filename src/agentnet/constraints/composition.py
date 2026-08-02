"""Composite constraint implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentnet.constraints.base import (
    Constraint,
    ConstraintDescriptor,
    ConstraintKind,
    ConstraintResult,
)
from agentnet.core import AgentNetConfigurationError


class CompositeConstraint(Constraint):
    """Base class for constraints composed from child constraints."""

    operator = "composite"

    def __init__(
        self,
        *constraints: Constraint,
        name: str | None = None,
        description: str | None = None,
        kind: ConstraintKind | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not constraints:
            raise AgentNetConfigurationError(
                "Composite constraints require at least one child constraint"
            )
        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise AgentNetConfigurationError(
                    "Composite constraints can only contain Constraint instances"
                )
        self.constraints = tuple(constraints)
        super().__init__(
            name or _default_composite_name(self.operator, self.constraints),
            description=description,
            kind=kind or _infer_constraint_kind(self.constraints),
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        return self.evaluate(candidate, context).passed

    def _evaluate_children(
        self, candidate: Any, context: Any | None
    ) -> tuple[ConstraintResult, ...]:
        return tuple(
            constraint.evaluate(candidate, context) for constraint in self.constraints
        )

    def _result(
        self,
        *,
        passed: bool,
        child_results: tuple[ConstraintResult, ...],
        kind: ConstraintKind | None = None,
    ) -> ConstraintResult:
        return ConstraintResult(
            constraint=self.name,
            passed=passed,
            kind=kind or self.kind,
            message=None if passed else f"Constraint {self.name!r} failed",
            metadata={
                "operator": self.operator,
                "results": [result.to_dict() for result in child_results],
            },
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={"operator": self.operator},
            children=tuple(
                constraint.to_descriptor() for constraint in self.constraints
            ),
        )


class AndConstraint(CompositeConstraint):
    """Constraint that passes when every child constraint passes."""

    operator = "and"

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        child_results = self._evaluate_children(candidate, context)
        passed = all(result.passed for result in child_results)
        return self._result(
            passed=passed,
            child_results=child_results,
            kind=_failed_child_kind(child_results) if not passed else None,
        )


class OrConstraint(CompositeConstraint):
    """Constraint that passes when any child constraint passes."""

    operator = "or"

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        child_results = self._evaluate_children(candidate, context)
        passed = any(result.passed for result in child_results)
        return self._result(
            passed=passed,
            child_results=child_results,
            kind=_failed_child_kind(child_results) if not passed else None,
        )


class NotConstraint(Constraint):
    """Constraint that inverts one child constraint."""

    def __init__(
        self,
        constraint: Constraint,
        *,
        name: str | None = None,
        description: str | None = None,
        kind: ConstraintKind | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not isinstance(constraint, Constraint):
            raise AgentNetConfigurationError(
                "NotConstraint can only wrap a Constraint instance"
            )
        self.constraint = constraint
        super().__init__(
            name or f"not({constraint.name})",
            description=description,
            kind=kind or constraint.kind,
            metadata=metadata,
            version=version,
        )

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        return self.evaluate(candidate, context).passed

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        child_result = self.constraint.evaluate(candidate, context)
        passed = not child_result.passed
        return ConstraintResult(
            constraint=self.name,
            passed=passed,
            kind=self.kind,
            message=None if passed else f"Constraint {self.name!r} failed",
            metadata={
                "operator": "not",
                "result": child_result.to_dict(),
            },
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
            parameters={"operator": "not"},
            children=(self.constraint.to_descriptor(),),
        )


def _default_composite_name(operator: str, constraints: tuple[Constraint, ...]) -> str:
    child_names = ", ".join(constraint.name for constraint in constraints)
    return f"{operator}({child_names})"


def _infer_constraint_kind(constraints: tuple[Constraint, ...]) -> ConstraintKind:
    if any(constraint.is_hard for constraint in constraints):
        return ConstraintKind.HARD
    return ConstraintKind.SOFT


def _failed_child_kind(results: tuple[ConstraintResult, ...]) -> ConstraintKind:
    failed_results = tuple(result for result in results if not result.passed)
    if any(result.is_hard for result in failed_results):
        return ConstraintKind.HARD
    return ConstraintKind.SOFT
