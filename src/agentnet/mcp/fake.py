"""In-memory fake MCP server for tests and local development."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import anyio

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.mcp.registry import MCPRegistry
from agentnet.mcp.server import MCPServer, MCPToolDescriptor
from agentnet.tools import ToolSpec


class FakeMCPServer:
    """Deterministic in-memory MCP server with explicit registered tools."""

    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str] | None = None,
        enabled: bool = True,
        metadata: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("Fake MCP server name cannot be empty")
        self.name = name
        self.command = tuple(command or ("fake-mcp", name))
        self.enabled = enabled
        self.metadata = dict(metadata or {})
        self.env = {str(key): str(value) for key, value in (env or {}).items()}
        self._specs: dict[str, ToolSpec] = {}
        self._tools: dict[str, Callable[..., Any]] = {}
        self._descriptors: dict[str, MCPToolDescriptor] = {}
        self._call_log: list[dict[str, Any]] = []

    @property
    def server(self) -> MCPServer:
        return MCPServer(
            name=self.name,
            command=self.command,
            enabled=self.enabled,
            metadata=self.metadata,
            env=self.env,
            tools=self._descriptors.values(),
        )

    @property
    def call_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "arguments": call["arguments"].copy(),
                "server": call["server"],
                "tool": call["tool"],
            }
            for call in self._call_log
        )

    def register_tool(
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
            raise AgentNetConfigurationError(
                f"Fake MCP tool {name!r} is already registered"
            )
        if not callable(implementation):
            raise AgentNetConfigurationError(
                f"Fake MCP tool {name!r} implementation must be callable"
            )

        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            metadata=metadata,
            output_schema=output_schema,
            side_effect=side_effect,
        )
        descriptor_metadata = spec.metadata.copy()
        if side_effect:
            descriptor_metadata["side_effect"] = True
        descriptor = MCPToolDescriptor(
            name,
            description=description,
            metadata=descriptor_metadata,
        )
        self._specs[name] = spec
        self._tools[name] = implementation
        self._descriptors[name] = descriptor
        return spec

    async def acall_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        if server_name != self.name:
            raise AgentNetValidationError(
                f"Fake MCP server {self.name!r} cannot serve server {server_name!r}"
            )
        try:
            spec = self._specs[tool_name]
            implementation = self._tools[tool_name]
        except KeyError as exc:
            raise AgentNetValidationError(f"Unknown MCP tool {tool_name!r}") from exc

        arguments_copy = dict(arguments or {})
        spec.validate_input(arguments_copy)
        self._call_log.append(
            {
                "arguments": arguments_copy.copy(),
                "server": server_name,
                "tool": tool_name,
            }
        )
        result = implementation(**arguments_copy)
        if inspect.isawaitable(result):
            result = await result
        return spec.validate_output(result)

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        async def _run() -> Any:
            return await self.acall_tool(server_name, tool_name, arguments)

        return anyio.run(_run)

    def to_registry(self, allow_tools: Iterable[str] | None = None) -> MCPRegistry:
        registry = MCPRegistry()
        registry.register(self.server, allow_tools=allow_tools)
        return registry
