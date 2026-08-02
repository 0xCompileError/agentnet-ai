"""Route-based graph execution."""

from collections.abc import Mapping
from typing import Any

from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetExecutionError,
    Module,
)


class Router(Module):
    """Route input to one child module based on a router module's output."""

    def __init__(
        self,
        *,
        router: Module,
        routes: Mapping[str, Module],
        fallback: Module | None = None,
        name: str = "router",
    ) -> None:
        if not routes:
            raise AgentNetConfigurationError("Router requires at least one route")
        super().__init__(name)
        self.router = router
        self.routes = dict(routes)
        self.fallback = fallback

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        route_key = str(await self.router.arun(input, context)).strip()
        route = self.routes.get(route_key, self.fallback)
        if route is None:
            raise AgentNetExecutionError(
                f"Router {self.name!r} selected unknown route {route_key!r}"
            )
        return await route.arun(input, context)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["fallback"] = self.fallback.state_dict() if self.fallback is not None else None
        state["router"] = self.router.state_dict()
        state["routes"] = {key: route.state_dict() for key, route in self.routes.items()}
        return state
