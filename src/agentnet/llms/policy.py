"""LLM selection policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError
from agentnet.llms.model_ref import ModelRef
from agentnet.policies import RetryPolicy


@dataclass(frozen=True, slots=True, init=False)
class LLMPolicy:
    """Ordered model candidate policy."""

    candidates: tuple[str, ...]
    retry_policy: RetryPolicy | None

    def __init__(
        self,
        candidates: Sequence[ModelRef | str],
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        candidate_aliases = tuple(_candidate_alias(candidate) for candidate in candidates)
        if not candidate_aliases:
            raise AgentNetConfigurationError("LLMPolicy requires at least one candidate")
        if len(set(candidate_aliases)) != len(candidate_aliases):
            raise AgentNetConfigurationError("LLMPolicy candidates must be unique")

        object.__setattr__(self, "candidates", candidate_aliases)
        object.__setattr__(self, "retry_policy", retry_policy)

    @property
    def primary(self) -> str:
        return self.candidates[0]

    @property
    def fallbacks(self) -> tuple[str, ...]:
        return self.candidates[1:]

    def with_fallback_order(self, fallbacks: Sequence[ModelRef | str]) -> Self:
        fallback_aliases = tuple(_candidate_alias(fallback) for fallback in fallbacks)
        if len(set(fallback_aliases)) != len(fallback_aliases):
            raise AgentNetConfigurationError("LLMPolicy fallback order cannot include duplicates")
        if set(fallback_aliases) != set(self.fallbacks):
            raise AgentNetConfigurationError(
                "LLMPolicy fallback order must include the existing fallback candidates"
            )

        return type(self)(
            candidates=(self.primary, *fallback_aliases),
            retry_policy=self.retry_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "retry_policy": (
                None if self.retry_policy is None else self.retry_policy.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, policy: dict[str, Any]) -> Self:
        retry_policy = policy.get("retry_policy")
        return cls(
            candidates=policy["candidates"],
            retry_policy=(
                None if retry_policy is None else RetryPolicy.from_dict(retry_policy)
            ),
        )


def _candidate_alias(candidate: ModelRef | str) -> str:
    alias = candidate.alias if isinstance(candidate, ModelRef) else str(candidate)
    if not alias:
        raise AgentNetConfigurationError("LLMPolicy candidate aliases cannot be empty")
    return alias
