"""Evaluation result and objective framework primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """Failure reported by an evaluation objective."""

    objective: str
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "metadata": dict(self.metadata),
            "objective": self.objective,
        }

    @classmethod
    def from_dict(cls, failure: Mapping[str, Any]) -> Self:
        return cls(
            objective=str(failure["objective"]),
            message=str(failure["message"]),
            metadata=dict(failure.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Result of evaluating one candidate against one or more objectives."""

    score: float
    passed: bool
    failures: Sequence[EvaluationFailure] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "failures": [failure.to_dict() for failure in self.failures],
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, result: Mapping[str, Any]) -> Self:
        return cls(
            score=float(result["score"]),
            passed=bool(result["passed"]),
            failures=tuple(
                EvaluationFailure.from_dict(failure)
                for failure in result.get("failures", ())
            ),
            metrics={
                str(key): float(value)
                for key, value in dict(result.get("metrics", {})).items()
            },
            metadata=dict(result.get("metadata", {})),
        )


class Objective:
    """Base class for objectives that score evaluation candidates."""

    def __init__(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
        weight: float = 1.0,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("Objective name cannot be empty")
        if weight < 0:
            raise AgentNetConfigurationError("Objective weight cannot be negative")
        version = str(version)
        if not version:
            raise AgentNetConfigurationError("Objective version cannot be empty")

        self.name = name
        self.description = description
        self.metadata = dict(metadata or {})
        self.version = version
        self.weight = float(weight)

    def __add__(self, other: Objective | ObjectiveSuite) -> ObjectiveSuite:
        if isinstance(other, ObjectiveSuite):
            return ObjectiveSuite((self, *other.objectives))
        if isinstance(other, Objective):
            return ObjectiveSuite((self, other))
        raise AgentNetConfigurationError("Objectives can only compose with objectives")

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "type": self.__class__.__name__,
            "version": self.version,
            "weight": self.weight,
        }


class ObjectiveSuite(Objective):
    """Composable collection of objectives evaluated as one objective."""

    def __init__(self, objectives: Iterable[Objective]) -> None:
        objective_tuple = tuple(objectives)
        if not objective_tuple:
            raise AgentNetConfigurationError("ObjectiveSuite requires at least one objective")
        for objective in objective_tuple:
            if not isinstance(objective, Objective):
                raise AgentNetConfigurationError(
                    "ObjectiveSuite entries must be Objective instances"
                )
        super().__init__("objective_suite")
        self.objectives = objective_tuple

    def __add__(self, other: Objective | ObjectiveSuite) -> ObjectiveSuite:
        if isinstance(other, ObjectiveSuite):
            return ObjectiveSuite((*self.objectives, *other.objectives))
        if isinstance(other, Objective):
            return ObjectiveSuite((*self.objectives, other))
        raise AgentNetConfigurationError("Objectives can only compose with objectives")

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        results = tuple(
            objective.evaluate(candidate, context) for objective in self.objectives
        )
        return aggregate_evaluation_results(
            results,
            weights=[objective.weight for objective in self.objectives],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["objectives"] = [objective.to_dict() for objective in self.objectives]
        return payload


def aggregate_evaluation_results(
    results: Iterable[EvaluationResult],
    *,
    weights: Iterable[float] | None = None,
) -> EvaluationResult:
    """Aggregate objective results with a weighted mean score."""

    result_tuple = tuple(results)
    if not result_tuple:
        raise AgentNetConfigurationError("At least one evaluation result is required")

    weight_tuple = (
        tuple(float(weight) for weight in weights)
        if weights is not None
        else tuple(1.0 for _ in result_tuple)
    )
    if len(weight_tuple) != len(result_tuple):
        raise AgentNetConfigurationError(
            "Evaluation aggregation weights must match result count"
        )
    if any(weight < 0 for weight in weight_tuple):
        raise AgentNetConfigurationError("Evaluation aggregation weights cannot be negative")
    total_weight = sum(weight_tuple)
    if total_weight <= 0:
        raise AgentNetConfigurationError("Evaluation aggregation weights must be positive")

    score = sum(
        result.score * weight
        for result, weight in zip(result_tuple, weight_tuple, strict=True)
    ) / total_weight
    failures = tuple(
        failure for result in result_tuple for failure in result.failures
    )
    metrics: dict[str, float] = {}
    for result in result_tuple:
        metrics.update(result.metrics)
    metrics["objective_count"] = float(len(result_tuple))

    return EvaluationResult(
        score=score,
        passed=all(result.passed for result in result_tuple),
        failures=failures,
        metrics=metrics,
        metadata={
            "evaluated_objectives": len(result_tuple),
            "failed_objectives": sum(1 for result in result_tuple if not result.passed),
        },
    )
