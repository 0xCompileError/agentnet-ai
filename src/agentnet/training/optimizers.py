"""Training optimizers for model policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.llms import LLMPolicy
from agentnet.mcp._security import validate_safe_metadata
from agentnet.policies import RetryPolicy


@dataclass(frozen=True, slots=True)
class FallbackOptimizationResult:
    """Best LLM fallback ordering selected by training."""

    policy: LLMPolicy
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="FallbackOptimizationResult")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", metadata)

    @property
    def candidate(self) -> LLMPolicy:
        return self.policy


class FallbackOptimizer:
    """Search fallback model orderings for an existing policy primary."""

    def __init__(
        self,
        *,
        max_candidates: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_candidates is not None and max_candidates < 1:
            raise AgentNetConfigurationError(
                "FallbackOptimizer max_candidates must be at least 1"
            )
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="FallbackOptimizer")
        self.max_candidates = max_candidates
        self.metadata = metadata_copy

    def optimize(
        self,
        policy: LLMPolicy,
        *,
        scorer: Callable[[LLMPolicy], float],
    ) -> FallbackOptimizationResult:
        best: FallbackOptimizationResult | None = None
        evaluated_candidates = 0

        for candidate in self._candidates(policy):
            if self.max_candidates is not None and evaluated_candidates >= self.max_candidates:
                break
            evaluated_candidates += 1
            score = float(scorer(candidate))
            result = FallbackOptimizationResult(
                policy=candidate,
                score=score,
                metadata={
                    **self.metadata,
                    "evaluated_candidates": evaluated_candidates,
                },
            )
            if best is None or result.score > best.score:
                best = result

        if best is None:
            raise AgentNetValidationError("No fallback policy candidate was evaluated")

        best.metadata["evaluated_candidates"] = evaluated_candidates
        return best

    def _candidates(self, policy: LLMPolicy) -> Iterable[LLMPolicy]:
        if not policy.fallbacks:
            yield policy
            return

        for fallback_order in permutations(policy.fallbacks):
            yield policy.with_fallback_order(fallback_order)


@dataclass(frozen=True, slots=True)
class RetryPolicyOptimizationResult:
    """Best retry policy selected by training."""

    policy: RetryPolicy
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="RetryPolicyOptimizationResult")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", metadata)

    @property
    def candidate(self) -> RetryPolicy:
        return self.policy


class RetryPolicyOptimizer:
    """Select the highest-scoring retry policy from explicit candidates."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="RetryPolicyOptimizer")
        self.metadata = metadata_copy

    def optimize(
        self,
        candidates: Iterable[RetryPolicy],
        *,
        scorer: Callable[[RetryPolicy], float],
    ) -> RetryPolicyOptimizationResult:
        best: RetryPolicyOptimizationResult | None = None
        evaluated_candidates = 0

        for policy in candidates:
            if not isinstance(policy, RetryPolicy):
                raise AgentNetConfigurationError(
                    "RetryPolicyOptimizer candidates must be RetryPolicy instances"
                )
            evaluated_candidates += 1
            score = float(scorer(policy))
            result = RetryPolicyOptimizationResult(
                policy=policy,
                score=score,
                metadata={
                    **self.metadata,
                    "evaluated_candidates": evaluated_candidates,
                },
            )
            if best is None or result.score > best.score:
                best = result

        if best is None:
            raise AgentNetValidationError("No retry policy candidate was provided")

        best.metadata["evaluated_candidates"] = evaluated_candidates
        return best
