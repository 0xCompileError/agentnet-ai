"""In-memory tool registry."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import anyio

from agentnet.core import AgentNetConfigurationError
from agentnet.tools.specs import ToolSpec


class ToolRegistry:
    """Register tool implementations alongside serializable descriptors."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._tools: dict[str, Callable[..., Any]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def register(
        self,
        name: str,
        implementation: Callable[..., Any],
        *,
        description: str | None = None,
        input_schema: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        output_schema: Any | None = None,
        side_effect: bool = False,
    ) -> ToolSpec:
        if name in self._specs:
            raise AgentNetConfigurationError(f"Tool {name!r} is already registered")
        if not callable(implementation):
            raise AgentNetConfigurationError(f"Tool {name!r} implementation must be callable")

        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            metadata=metadata,
            output_schema=output_schema,
            side_effect=side_effect,
        )
        self._specs[name] = spec
        self._tools[name] = implementation
        return spec

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(f"Unknown tool {name!r}") from exc

    def get_spec(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(f"Unknown tool {name!r}") from exc

    async def aexecute(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        agent: Any | None = None,
        context: Any | None = None,
    ) -> Any:
        if agent is not None and hasattr(agent, "require_tool"):
            agent.require_tool(name)

        spec = self.get_spec(name)
        implementation = self.get(name)
        arguments_copy = dict(arguments or {})
        spec.validate_input(arguments_copy)

        _record_tool_event(context, agent=agent, spec=spec, event_type="tool.called")
        if getattr(implementation, "_agentnet_accepts_context", False):
            result = implementation(
                _agentnet_agent=agent,
                _agentnet_context=context,
                **arguments_copy,
            )
        else:
            result = implementation(**arguments_copy)
        if inspect.isawaitable(result):
            result = await result

        validated = spec.validate_output(result)
        _record_tool_event(context, agent=agent, spec=spec, event_type="tool.completed")
        return validated

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        agent: Any | None = None,
        context: Any | None = None,
    ) -> Any:
        async def _run() -> Any:
            return await self.aexecute(name, arguments, agent=agent, context=context)

        return anyio.run(_run)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [spec.to_dict() for spec in self._specs.values()],
        }


def _record_tool_event(
    context: Any | None,
    *,
    agent: Any | None,
    spec: ToolSpec,
    event_type: str,
) -> None:
    if context is None or not hasattr(context, "metadata"):
        return

    tool_events = context.metadata.setdefault("tool_events", [])
    if not isinstance(tool_events, list):
        return

    tool_events.append(
        {
            "agent": getattr(agent, "name", None),
            "side_effect": spec.side_effect,
            "tool": spec.name,
            "type": event_type,
        }
    )
