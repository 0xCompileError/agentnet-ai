"""Constraint-aware communication protocol optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentnet.constraints import (
    Constraint,
    ConstraintResult,
    validate_training_constraints,
)
from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.interfaces import Interface, Representation


@dataclass(frozen=True, slots=True)
class CommunicationProtocol:
    """Direct communication protocol candidate for an interface pair."""

    source: Interface
    target: Interface
    representation: str
    representation_object: Representation
    mode: str = "direct"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mode:
            raise AgentNetConfigurationError("CommunicationProtocol mode cannot be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def identifier(self) -> str:
        return self.representation


@dataclass(frozen=True, slots=True)
class CommunicationProtocolOptimizationResult:
    """Best communication protocol selected for an interface pair."""

    protocol: CommunicationProtocol
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
    def candidate(self) -> CommunicationProtocol:
        return self.protocol


class CommunicationProtocolOptimizer:
    """Score direct communication protocols between compatible representations."""

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
        target: Interface,
        *,
        scorer: Callable[[CommunicationProtocol], float],
    ) -> CommunicationProtocolOptimizationResult:
        best: CommunicationProtocolOptimizationResult | None = None
        evaluated_candidates = 0
        rejected_candidates = 0

        for protocol in _direct_protocols(source, target):
            candidate_metadata: dict[str, Any] = {}
            try:
                constraint_results = validate_training_constraints(
                    protocol,
                    self.constraints,
                    metadata=candidate_metadata,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            evaluated_candidates += 1
            score = float(scorer(protocol))
            result = CommunicationProtocolOptimizationResult(
                protocol=protocol,
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
                "No communication protocol candidate satisfied hard constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        return best


def _direct_protocols(
    source: Interface,
    target: Interface,
) -> tuple[CommunicationProtocol, ...]:
    source_identifiers = source.representation_identifiers
    target_identifiers = target.representation_identifiers
    if not source_identifiers and not target_identifiers:
        return ()

    if source_identifiers and target_identifiers:
        target_set = set(target_identifiers)
        representations = tuple(
            source.get_representation(identifier)
            for identifier in source_identifiers
            if identifier in target_set
        )
    else:
        representations = source.representations or target.representations

    return tuple(
        CommunicationProtocol(
            source=source,
            target=target,
            representation=representation.identifier,
            representation_object=representation,
        )
        for representation in representations
    )
