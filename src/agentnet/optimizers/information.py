"""Constraint-aware downstream information transfer optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentnet.constraints import (
    Constraint,
    ConstraintResult,
    validate_training_constraints,
)
from agentnet.core import AgentNetValidationError
from agentnet.interfaces import Interface


@dataclass(frozen=True, slots=True)
class _InformationTransferCandidate:
    target: Interface
    value: Any
    representation: str | None


@dataclass(frozen=True, slots=True)
class InformationTransferOptimizationResult:
    """Best downstream payload selected for a target interface."""

    target: Interface
    value: Any
    representation: str | None
    score: float
    constraint_results: tuple[ConstraintResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint_results",
            tuple(self.constraint_results),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def candidate(self) -> Any:
        return self.value


class InformationTransferOptimizer:
    """Select valid downstream payload candidates for a target interface."""

    def __init__(
        self,
        *,
        constraints: Iterable[Constraint] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.constraints = tuple(constraints or ())
        self.metadata = dict(metadata or {})

    def optimize(
        self,
        candidates: Iterable[Any],
        *,
        target: Interface,
        scorer: Callable[[Any], float],
        representation: str | None = None,
    ) -> InformationTransferOptimizationResult:
        best: InformationTransferOptimizationResult | None = None
        evaluated_candidates = 0
        rejected_candidates = 0

        for value in candidates:
            try:
                validated = target.validate(
                    value,
                    label="candidate",
                    representation=representation,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            candidate = _InformationTransferCandidate(
                target=target,
                value=validated,
                representation=representation,
            )
            candidate_metadata: dict[str, Any] = {}
            try:
                constraint_results = validate_training_constraints(
                    candidate,
                    self.constraints,
                    metadata=candidate_metadata,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            evaluated_candidates += 1
            score = float(scorer(validated))
            result = InformationTransferOptimizationResult(
                target=target,
                value=validated,
                representation=representation,
                score=score,
                constraint_results=constraint_results,
                metadata={
                    **self.metadata,
                    "evaluated_candidates": evaluated_candidates,
                    "rejected_candidates": rejected_candidates,
                    "training_constraint_results": candidate_metadata.get(
                        "training_constraint_results",
                        [],
                    ),
                },
            )
            if best is None or result.score > best.score:
                best = result

        if best is None:
            raise AgentNetValidationError(
                "No information transfer candidate satisfied hard constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        return best
