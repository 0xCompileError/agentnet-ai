"""Constraint-aware prompt candidate optimization."""

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
class PromptOptimizationResult:
    """Best prompt selected by prompt optimization."""

    prompt: str
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
    def candidate(self) -> str:
        return self.prompt


class PromptOptimizer:
    """Search explicit prompt candidates within training constraints."""

    def __init__(
        self,
        *,
        constraints: Iterable[Constraint] | None = None,
        allow_empty: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.constraints = tuple(constraints or ())
        self.allow_empty = bool(allow_empty)
        self.metadata = dict(metadata or {})

    def optimize(
        self,
        candidates: Iterable[str],
        *,
        scorer: Callable[[str], float],
    ) -> PromptOptimizationResult:
        best: PromptOptimizationResult | None = None
        evaluated_candidates = 0
        rejected_candidates = 0

        for prompt in candidates:
            if not isinstance(prompt, str) or (
                not self.allow_empty and not prompt.strip()
            ):
                rejected_candidates += 1
                continue

            candidate_metadata: dict[str, Any] = {}
            try:
                constraint_results = validate_training_constraints(
                    prompt,
                    self.constraints,
                    metadata=candidate_metadata,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            evaluated_candidates += 1
            score = float(scorer(prompt))
            result = PromptOptimizationResult(
                prompt=prompt,
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
                "No prompt candidate satisfied hard constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        return best
