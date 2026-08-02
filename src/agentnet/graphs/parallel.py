"""Parallel graph execution."""

from typing import Any

import anyio

from agentnet.core import AgentNetConfigurationError, Module


class Parallel(Module):
    """Run child modules against the same input and return ordered branch outputs."""

    def __init__(
        self,
        *modules: Module,
        reducer: Module | None = None,
        name: str = "parallel",
    ) -> None:
        if not modules:
            raise AgentNetConfigurationError("Parallel requires at least one module")
        super().__init__(name)
        self.modules = tuple(modules)
        self.reducer = reducer

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        outputs: list[Any] = [None] * len(self.modules)

        async def run_module(index: int, module: Module) -> None:
            outputs[index] = await module.arun(input, context)

        async with anyio.create_task_group() as task_group:
            for index, module in enumerate(self.modules):
                task_group.start_soon(run_module, index, module)
        branch_outputs = tuple(outputs)
        if self.reducer is not None:
            return await self.reducer.arun(branch_outputs, context)
        return branch_outputs

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["modules"] = [module.state_dict() for module in self.modules]
        state["reducer"] = self.reducer.state_dict() if self.reducer is not None else None
        return state
