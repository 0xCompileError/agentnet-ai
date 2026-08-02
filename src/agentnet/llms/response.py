"""Chat response values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from agentnet.llms.model_ref import ModelRef


@dataclass(frozen=True, slots=True, init=False)
class ChatResponse:
    """Provider-agnostic chat completion response."""

    content: str
    model: ModelRef
    usage: dict[str, int]
    finish_reason: str | None
    metadata: dict[str, Any]

    def __init__(
        self,
        content: str,
        model: ModelRef | str,
        usage: Mapping[str, int] | None = None,
        finish_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "model", model if isinstance(model, ModelRef) else ModelRef(model))
        object.__setattr__(self, "usage", dict(usage or {}))
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata.copy(),
            "model": self.model.to_dict(),
            "usage": self.usage.copy(),
        }

    @classmethod
    def from_dict(cls, response: dict[str, Any]) -> Self:
        return cls(
            content=str(response["content"]),
            model=ModelRef.from_dict(response["model"]),
            usage=response.get("usage", {}),
            finish_reason=response.get("finish_reason"),
            metadata=response.get("metadata", {}),
        )
