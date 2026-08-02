"""Generic constraint-aware optimizer primitives."""

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


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Best candidate selected by an optimizer."""

    candidate: Any
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


class ConstraintAwareOptimizer:
    """Score candidates while respecting hard training constraints."""

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
        scorer: Callable[[Any], float],
    ) -> OptimizationResult:
        best: OptimizationResult | None = None
        evaluated_candidates = 0
        rejected_candidates = 0

        for candidate in candidates:
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
            score = float(scorer(candidate))
            result = OptimizationResult(
                candidate=candidate,
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
                "No candidate satisfied hard optimizer constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        return best
