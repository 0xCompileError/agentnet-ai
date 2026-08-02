"""Constraint-aware interface compatibility optimization."""

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
from agentnet.interfaces import (
    Interface,
    InterfaceCompatibility,
    validate_interface_compatibility,
)


@dataclass(frozen=True, slots=True)
class _InterfaceCompatibilityCandidate:
    source: Interface
    target: Interface
    compatibility: InterfaceCompatibility
    representation: str | None


@dataclass(frozen=True, slots=True)
class InterfaceCompatibilityOptimizationResult:
    """Best compatible target interface selected for a source interface."""

    compatibility: InterfaceCompatibility
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
    def source(self) -> Interface:
        return self.compatibility.source

    @property
    def target(self) -> Interface:
        return self.compatibility.target

    @property
    def representation(self) -> str | None:
        return self.compatibility.negotiation.identifier

    @property
    def candidate(self) -> InterfaceCompatibility:
        return self.compatibility


class InterfaceCompatibilityOptimizer:
    """Select the best compatible target interface for a source interface."""

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
        source: Interface,
        targets: Iterable[Interface],
        *,
        scorer: Callable[[InterfaceCompatibility], float],
        preferred: Iterable[str] | None = None,
    ) -> InterfaceCompatibilityOptimizationResult:
        best: InterfaceCompatibilityOptimizationResult | None = None
        evaluated_candidates = 0
        rejected_candidates = 0

        for target in targets:
            try:
                compatibility = validate_interface_compatibility(
                    source,
                    target,
                    preferred=preferred,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            candidate = _InterfaceCompatibilityCandidate(
                source=source,
                target=target,
                compatibility=compatibility,
                representation=compatibility.negotiation.identifier,
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
            score = float(scorer(compatibility))
            result = InterfaceCompatibilityOptimizationResult(
                compatibility=compatibility,
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
                "No interface compatibility candidate satisfied hard constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        return best
