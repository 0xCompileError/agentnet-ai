"""Descriptor-only MCP server registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.mcp.server import MCPServer, MCPToolDescriptor


class MCPRegistry:
    """Register MCP server descriptors and resolve declared MCP tools."""

    def __init__(self, servers: Iterable[MCPServer] | None = None) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._allowlists: dict[str, tuple[str, ...]] = {}
        for server in servers or ():
            self.register(server)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._servers)

    @property
    def server_names(self) -> tuple[str, ...]:
        return self.names

    @property
    def qualified_tool_names(self) -> tuple[str, ...]:
        return tuple(
            f"{server.name}.{tool_name}"
            for server in self._servers.values()
            for tool_name in self._allowlists[server.name]
        )

    def register(
        self,
        server: MCPServer,
        *,
        allow_tools: Iterable[str] | None = None,
    ) -> MCPServer:
        if not isinstance(server, MCPServer):
            raise AgentNetConfigurationError(
                "MCP registry entries must be MCPServer instances"
            )
        if server.name in self._servers:
            raise AgentNetConfigurationError(
                f"MCP server {server.name!r} is already registered"
            )
        allowed_tools = _normalize_allowlist(server, allow_tools)
        self._servers[server.name] = server
        self._allowlists[server.name] = allowed_tools
        return server

    def get_server(self, name: str) -> MCPServer:
        try:
            return self._servers[name]
        except KeyError as exc:
            raise AgentNetConfigurationError(f"Unknown MCP server {name!r}") from exc

    def allowed_tools(self, server_name: str) -> tuple[str, ...]:
        self.get_server(server_name)
        return self._allowlists[server_name]

    def is_tool_allowed(self, qualified_name: str) -> bool:
        try:
            server_name, tool_name = _split_qualified_tool_name(qualified_name)
            self.get_server(server_name)
        except (AgentNetConfigurationError, AgentNetValidationError):
            return False
        return tool_name in self._allowlists[server_name]

    def get_tool(self, qualified_name: str) -> MCPToolDescriptor:
        server_name, tool_name = _split_qualified_tool_name(qualified_name)
        try:
            server = self._servers[server_name]
        except KeyError as exc:
            raise AgentNetValidationError(
                f"Unknown MCP server {server_name!r}"
            ) from exc
        tool = server.get_tool(tool_name)
        if tool_name not in self._allowlists[server_name]:
            raise AgentNetValidationError(
                f"MCP tool {qualified_name!r} is not allowed"
            )
        return tool

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "servers": [server.to_dict() for server in self._servers.values()],
        }
        restricted = {
            name: list(allowlist)
            for name, allowlist in self._allowlists.items()
            if allowlist != self._servers[name].tool_names
        }
        if restricted:
            payload["allowlists"] = restricted
        return payload


def _split_qualified_tool_name(qualified_name: str) -> tuple[str, str]:
    server_name, separator, tool_name = qualified_name.partition(".")
    if not separator or not server_name or not tool_name:
        raise AgentNetValidationError(
            "MCP tool name must be qualified as 'server.tool'"
        )
    return server_name, tool_name


def _normalize_allowlist(
    server: MCPServer,
    allow_tools: Iterable[str] | None,
) -> tuple[str, ...]:
    if allow_tools is None:
        return server.tool_names

    allowed = tuple(str(tool_name) for tool_name in allow_tools)
    if any(not tool_name for tool_name in allowed):
        raise AgentNetConfigurationError("MCP allowlists cannot include empty tools")
    if len(set(allowed)) != len(allowed):
        raise AgentNetConfigurationError("MCP allowlists cannot include duplicates")

    declared = set(server.tool_names)
    unknown = [tool_name for tool_name in allowed if tool_name not in declared]
    if unknown:
        raise AgentNetConfigurationError(
            f"Unknown MCP tool in allowlist for {server.name!r}: {', '.join(unknown)}"
        )
    return allowed
