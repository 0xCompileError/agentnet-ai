"""Run context primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Self

from agentnet.core.state import GraphState


@dataclass(slots=True, init=False)
class RunContext:
    """Context shared across one graph execution."""

    run_id: str
    graph_state: GraphState
    metadata: dict[str, Any]
    cancelled: bool

    def __init__(
        self,
        run_id: str,
        graph_state: GraphState | None = None,
        metadata: dict[str, Any] | None = None,
        cancelled: bool = False,
    ) -> None:
        self.run_id = run_id
        self.graph_state = graph_state if graph_state is not None else GraphState(run_id=run_id)
        self.metadata = metadata.copy() if metadata is not None else {}
        self.cancelled = cancelled

    def cancel(self) -> None:
        self.cancelled = True

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError

    def to_dict(self) -> dict[str, Any]:
        return {
            "cancelled": self.cancelled,
            "graph_state": self.graph_state.to_dict(),
            "metadata": self.metadata.copy(),
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, context: dict[str, Any]) -> Self:
        return cls(
            run_id=str(context["run_id"]),
            graph_state=GraphState.from_dict(context["graph_state"]),
            metadata=dict(context.get("metadata", {})),
            cancelled=bool(context.get("cancelled", False)),
        )
