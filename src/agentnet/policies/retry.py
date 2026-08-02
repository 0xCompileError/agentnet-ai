"""Retry policy configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError


@dataclass(frozen=True, slots=True, init=False)
class RetryPolicy:
    """Serializable retry policy configuration."""

    transport_retries: int
    quality_retries: int
    fallback_on: tuple[str, ...]
    backoff: str
    backoff_base_seconds: float
    max_total_attempts: int | None

    def __init__(
        self,
        transport_retries: int = 2,
        quality_retries: int = 1,
        fallback_on: Sequence[str] = ("timeout", "rate_limit", "api_error", "schema_failure"),
        backoff: str = "exponential",
        backoff_base_seconds: float = 0.0,
        max_total_attempts: int | None = None,
    ) -> None:
        if transport_retries < 0:
            raise AgentNetConfigurationError("RetryPolicy transport_retries cannot be negative")
        if quality_retries < 0:
            raise AgentNetConfigurationError("RetryPolicy quality_retries cannot be negative")

        fallback_reasons = tuple(fallback_on)
        if not fallback_reasons:
            raise AgentNetConfigurationError(
                "RetryPolicy fallback_on must include at least one reason"
            )
        if any(not reason for reason in fallback_reasons):
            raise AgentNetConfigurationError("RetryPolicy fallback_on cannot include empty reasons")

        if backoff not in {"none", "constant", "linear", "exponential"}:
            raise AgentNetConfigurationError(
                "RetryPolicy backoff must be one of: none, constant, linear, exponential"
            )
        if backoff_base_seconds < 0:
            raise AgentNetConfigurationError(
                "RetryPolicy backoff_base_seconds cannot be negative"
            )

        if max_total_attempts is not None and max_total_attempts < 1:
            raise AgentNetConfigurationError(
                "RetryPolicy max_total_attempts must be at least 1 when set"
            )

        object.__setattr__(self, "transport_retries", transport_retries)
        object.__setattr__(self, "quality_retries", quality_retries)
        object.__setattr__(self, "fallback_on", fallback_reasons)
        object.__setattr__(self, "backoff", backoff)
        object.__setattr__(self, "backoff_base_seconds", float(backoff_base_seconds))
        object.__setattr__(self, "max_total_attempts", max_total_attempts)

    def should_fallback(self, reason: str) -> bool:
        return reason in self.fallback_on

    def backoff_delay(self, retry_number: int) -> float:
        if retry_number < 1:
            raise AgentNetConfigurationError("RetryPolicy retry_number must be at least 1")

        if self.backoff == "none":
            return 0.0
        if self.backoff == "constant":
            return self.backoff_base_seconds
        if self.backoff == "linear":
            return self.backoff_base_seconds * retry_number
        return self.backoff_base_seconds * (2 ** (retry_number - 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backoff": self.backoff,
            "backoff_base_seconds": self.backoff_base_seconds,
            "fallback_on": list(self.fallback_on),
            "max_total_attempts": self.max_total_attempts,
            "quality_retries": self.quality_retries,
            "transport_retries": self.transport_retries,
        }

    @classmethod
    def from_dict(cls, policy: dict[str, Any]) -> Self:
        return cls(
            transport_retries=int(policy.get("transport_retries", 2)),
            quality_retries=int(policy.get("quality_retries", 1)),
            fallback_on=policy.get(
                "fallback_on",
                ("timeout", "rate_limit", "api_error", "schema_failure"),
            ),
            backoff=str(policy.get("backoff", "exponential")),
            backoff_base_seconds=float(policy.get("backoff_base_seconds", 0.0)),
            max_total_attempts=(
                None
                if policy.get("max_total_attempts") is None
                else int(policy["max_total_attempts"])
            ),
        )
