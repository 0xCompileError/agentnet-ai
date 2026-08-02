"""Constraint-aware translation strategy optimization."""

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
from agentnet.interfaces import (
    Interface,
    RepresentationTranslator,
    RepresentationTranslatorRegistry,
)


@dataclass(frozen=True, slots=True)
class TranslationStrategy:
    """One-hop strategy for passing or translating a representation."""

    source: Interface
    target: Interface
    source_representation: str
    target_representation: str
    translator: RepresentationTranslator | None = None
    mode: str = "identity"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mode:
            raise AgentNetConfigurationError("TranslationStrategy mode cannot be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def representation(self) -> str:
        return self.target_representation

    @property
    def identifier(self) -> str:
        return f"{self.source_representation}->{self.target_representation}"

    def translate(self, value: Any) -> Any:
        if self.translator is None:
            return value
        return self.translator.translate(value)


@dataclass(frozen=True, slots=True)
class TranslationStrategyOptimizationResult:
    """Best translation strategy selected for an interface pair."""

    strategy: TranslationStrategy
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
    def candidate(self) -> TranslationStrategy:
        return self.strategy


class TranslationStrategyOptimizer:
    """Score identity and registered one-hop translation strategies."""

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
        scorer: Callable[[TranslationStrategy], float],
        translator_registry: RepresentationTranslatorRegistry | None = None,
    ) -> TranslationStrategyOptimizationResult:
        best: TranslationStrategyOptimizationResult | None = None
        evaluated_candidates = 0
        rejected_candidates = 0

        for strategy in _translation_strategies(
            source,
            target,
            translator_registry or RepresentationTranslatorRegistry(),
        ):
            candidate_metadata: dict[str, Any] = {}
            try:
                constraint_results = validate_training_constraints(
                    strategy,
                    self.constraints,
                    metadata=candidate_metadata,
                )
            except AgentNetValidationError:
                rejected_candidates += 1
                continue

            evaluated_candidates += 1
            score = float(scorer(strategy))
            result = TranslationStrategyOptimizationResult(
                strategy=strategy,
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
                "No translation strategy candidate satisfied hard constraints"
            )

        best.metadata["evaluated_candidates"] = evaluated_candidates
        best.metadata["rejected_candidates"] = rejected_candidates
        return best


def _translation_strategies(
    source: Interface,
    target: Interface,
    translator_registry: RepresentationTranslatorRegistry,
) -> tuple[TranslationStrategy, ...]:
    strategies = [
        TranslationStrategy(
            source=source,
            target=target,
            source_representation=identifier,
            target_representation=identifier,
            mode="identity",
        )
        for identifier in _direct_identifiers(source, target)
    ]

    source_identifiers = set(source.representation_identifiers)
    target_identifiers = set(target.representation_identifiers)
    for translator in translator_registry.translators:
        if translator.source not in source_identifiers:
            continue
        if translator.target not in target_identifiers:
            continue
        strategies.append(
            TranslationStrategy(
                source=source,
                target=target,
                source_representation=translator.source,
                target_representation=translator.target,
                translator=translator,
                mode="translate",
            )
        )
    return tuple(strategies)


def _direct_identifiers(source: Interface, target: Interface) -> tuple[str, ...]:
    source_identifiers = source.representation_identifiers
    target_identifiers = target.representation_identifiers
    if source_identifiers and target_identifiers:
        target_set = set(target_identifiers)
        return tuple(identifier for identifier in source_identifiers if identifier in target_set)
    return source_identifiers or target_identifiers
