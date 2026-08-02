"""Reducer graph execution."""

from typing import Any

from agentnet.core import Module


class Reducer(Module):
    """Pass branch outputs to a reducer module."""

    def __init__(self, reducer: Module, name: str = "reducer") -> None:
        super().__init__(name)
        self.reducer = reducer

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        return await self.reducer.arun(input, context)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["reducer"] = self.reducer.state_dict()
        return state
