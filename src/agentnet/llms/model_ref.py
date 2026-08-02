"""Model reference values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Reference to a configured model candidate."""

    alias: str
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.alias:
            raise AgentNetConfigurationError("ModelRef alias cannot be empty")

    def __str__(self) -> str:
        return self.alias

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, model_ref: dict[str, Any]) -> Self:
        return cls(
            alias=str(model_ref["alias"]),
            provider=model_ref.get("provider"),
            model=model_ref.get("model"),
        )
