"""Execution result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.core.state import GraphState


@dataclass(slots=True)
class GraphResult:
    """Result of one graph execution."""

    output: Any
    graph_state: GraphState
    succeeded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_state": self.graph_state.to_dict(),
            "metadata": self.metadata.copy(),
            "output": self.output,
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, result: dict[str, Any]) -> Self:
        return cls(
            output=result.get("output"),
            graph_state=GraphState.from_dict(result["graph_state"]),
            succeeded=bool(result.get("succeeded", True)),
            metadata=dict(result.get("metadata", {})),
        )
