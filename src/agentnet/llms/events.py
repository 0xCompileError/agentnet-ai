"""Streaming chat events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from agentnet.llms.model_ref import ModelRef
from agentnet.llms.response import ChatResponse


@dataclass(frozen=True, slots=True, init=False)
class ChatEvent:
    """Incremental chat stream event."""

    delta: str
    model: ModelRef
    content: str | None
    usage: dict[str, int]
    finish_reason: str | None
    metadata: dict[str, Any]

    def __init__(
        self,
        *,
        delta: str,
        model: ModelRef | str,
        content: str | None = None,
        usage: Mapping[str, int] | None = None,
        finish_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "model", model if isinstance(model, ModelRef) else ModelRef(model))
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "usage", dict(usage or {}))
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "delta": self.delta,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata.copy(),
            "model": self.model.to_dict(),
            "usage": self.usage.copy(),
        }

    @classmethod
    def from_dict(cls, event: dict[str, Any]) -> Self:
        return cls(
            delta=str(event["delta"]),
            model=ModelRef.from_dict(event["model"]),
            content=event.get("content"),
            usage=event.get("usage", {}),
            finish_reason=event.get("finish_reason"),
            metadata=event.get("metadata", {}),
        )

    @classmethod
    def from_response(cls, response: ChatResponse) -> Self:
        return cls(
            content=response.content,
            delta=response.content,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
            model=response.model,
            usage=response.usage,
        )
