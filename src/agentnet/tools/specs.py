"""Serializable tool descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError
from agentnet.core.schema import validate_schema

_SECRET_KEY_PARTS = ("api_key", "password", "secret", "token")


@dataclass(frozen=True, slots=True, init=False)
class ToolSpec:
    """Serializable description of a tool without its implementation."""

    name: str
    description: str | None = None
    side_effect: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    input_schema: Any | None = None
    output_schema: Any | None = None

    def __init__(
        self,
        name: str,
        description: str | None = None,
        side_effect: bool = False,
        metadata: Mapping[str, Any] | None = None,
        input_schema: Any | None = None,
        output_schema: Any | None = None,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("ToolSpec name cannot be empty")
        metadata_copy = dict(metadata or {})
        _validate_metadata_keys(metadata_copy)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "side_effect", side_effect)
        object.__setattr__(self, "metadata", metadata_copy)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)

    def validate_input(self, value: Any) -> Any:
        return validate_schema(self.input_schema, value, label=f"tool input {self.name}")

    def validate_output(self, value: Any) -> Any:
        return validate_schema(self.output_schema, value, label=f"tool output {self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "side_effect": self.side_effect,
        }

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> Self:
        return cls(
            name=str(spec["name"]),
            description=spec.get("description"),
            metadata=spec.get("metadata", {}),
            side_effect=bool(spec.get("side_effect", False)),
        )


def _validate_metadata_keys(metadata: Mapping[str, Any]) -> None:
    for key in metadata:
        normalized = str(key).lower()
        if any(part in normalized for part in _SECRET_KEY_PARTS):
            raise AgentNetConfigurationError(
                f"ToolSpec metadata key {key!r} may serialize secrets"
            )
