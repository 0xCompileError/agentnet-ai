"""Base module abstraction."""

from collections.abc import Mapping
from typing import Any

import anyio


class Module:
    """Base class for executable AgentNet components."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        raise NotImplementedError

    def run(
        self,
        input: Any,
        context: Any | None = None,
        *,
        scheduler: Any | None = None,
        retry_policy: Any | None = None,
    ) -> Any:
        if scheduler is not None:
            result = scheduler.run(
                self,
                input,
                context,
                retry_policy=retry_policy,
            )
            return result.output
        return anyio.run(self.arun, input, context)

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.name = str(state["name"])
