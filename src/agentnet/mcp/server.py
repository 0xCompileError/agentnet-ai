"""Descriptor-only MCP server abstractions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.mcp._security import validate_safe_metadata


@dataclass(frozen=True, slots=True, init=False)
class MCPToolDescriptor:
    """Serializable MCP tool descriptor without executable implementation."""

    name: str
    description: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("MCP tool name cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="MCP tool descriptor")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "metadata": self.metadata.copy(),
            "name": self.name,
        }


@dataclass(frozen=True, slots=True, init=False)
class MCPServer:
    """Descriptor-only MCP server configuration.

    The server abstraction stores launch configuration but never executes it.
    Environment variables are retained for runtime setup and intentionally omitted
    from serialization to avoid leaking secrets.
    """

    name: str
    command: tuple[str, ...]
    tools: tuple[MCPToolDescriptor, ...]
    enabled: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str],
        tools: Iterable[MCPToolDescriptor] | None = None,
        enabled: bool = False,
        metadata: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("MCP server name cannot be empty")
        command_tuple = _normalize_command(command)
        tool_tuple = tuple(tools or ())
        _validate_tools(tool_tuple)
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="MCP server descriptor")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "command", command_tuple)
        object.__setattr__(self, "tools", tool_tuple)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "metadata", metadata_copy)
        object.__setattr__(
            self,
            "env",
            {str(key): str(value) for key, value in (env or {}).items()},
        )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    @property
    def qualified_tool_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}.{tool_name}" for tool_name in self.tool_names)

    def get_tool(self, name: str) -> MCPToolDescriptor:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise AgentNetValidationError(f"Unknown MCP tool {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "enabled": self.enabled,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "tools": [tool.to_dict() for tool in self.tools],
        }


def _normalize_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        raise AgentNetConfigurationError("MCP server command must be a sequence")
    command_tuple = tuple(str(part) for part in command)
    if not command_tuple:
        raise AgentNetConfigurationError("MCP server command cannot be empty")
    if any(not part for part in command_tuple):
        raise AgentNetConfigurationError("MCP server command parts cannot be empty")
    return command_tuple


def _validate_tools(tools: tuple[MCPToolDescriptor, ...]) -> None:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, MCPToolDescriptor):
            raise AgentNetConfigurationError(
                "MCP server tools must be MCPToolDescriptor instances"
            )
        if tool.name in names:
            raise AgentNetConfigurationError(
                f"Duplicate MCP tool descriptor {tool.name!r}"
            )
        names.add(tool.name)
