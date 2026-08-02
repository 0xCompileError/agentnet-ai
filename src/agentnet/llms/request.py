"""Chat request values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from agentnet.core import AgentNetValidationError
from agentnet.llms.model_ref import ModelRef


@dataclass(frozen=True, slots=True, init=False)
class ChatRequest:
    """Provider-agnostic chat completion request."""

    model: ModelRef
    messages: tuple[dict[str, str], ...]
    metadata: dict[str, Any]

    def __init__(
        self,
        model: ModelRef | str,
        messages: Sequence[Mapping[str, str]],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not messages:
            raise AgentNetValidationError("ChatRequest requires at least one message")

        object.__setattr__(self, "model", model if isinstance(model, ModelRef) else ModelRef(model))
        object.__setattr__(self, "messages", tuple(dict(message) for message in messages))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.copy() for message in self.messages],
            "metadata": self.metadata.copy(),
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, request: dict[str, Any]) -> Self:
        return cls(
            model=ModelRef.from_dict(request["model"]),
            messages=request["messages"],
            metadata=request.get("metadata", {}),
        )
