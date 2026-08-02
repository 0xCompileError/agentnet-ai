"""Sequential graph execution."""

from typing import Any

from agentnet.core import AgentNetConfigurationError, Module


class Sequential(Module):
    """Run child modules in order, passing each output to the next module."""

    def __init__(self, *modules: Module, name: str = "sequential") -> None:
        if not modules:
            raise AgentNetConfigurationError("Sequential requires at least one module")
        super().__init__(name)
        self.modules = tuple(modules)

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        output = input
        for module in self.modules:
            output = await module.arun(output, context)
        return output

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["modules"] = [module.state_dict() for module in self.modules]
        return state
