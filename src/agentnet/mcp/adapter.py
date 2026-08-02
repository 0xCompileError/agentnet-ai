"""Adapter from MCP descriptors to AgentNet tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import anyio

from agentnet.core import AgentNetConfigurationError
from agentnet.mcp.approvals import MCPApprovalStore
from agentnet.mcp.descriptors import hash_mcp_descriptor
from agentnet.mcp.registry import MCPRegistry
from agentnet.tools import ToolRegistry, ToolSpec


class MCPToolAdapter:
    """Expose approved MCP tool descriptors through AgentNet's tool registry."""

    def __init__(
        self,
        registry: MCPRegistry,
        client: Any,
        *,
        approvals: MCPApprovalStore | None = None,
        require_approval: bool = False,
    ) -> None:
        if not isinstance(registry, MCPRegistry):
            raise AgentNetConfigurationError(
                "MCPToolAdapter registry must be an MCPRegistry"
            )
        self.registry = registry
        self.client = client
        self.approvals = approvals
        self.require_approval = require_approval

    def to_tool_spec(self, qualified_name: str) -> ToolSpec:
        descriptor = self.registry.get_tool(qualified_name)
        server_name, tool_name = _split_qualified_name(qualified_name)
        metadata = descriptor.metadata.copy()
        side_effect = bool(metadata.pop("side_effect", False))
        metadata.update(
            {
                "mcp": True,
                "mcp_descriptor_hash": hash_mcp_descriptor(descriptor),
                "mcp_server": server_name,
                "mcp_tool": tool_name,
            }
        )
        return ToolSpec(
            name=qualified_name,
            description=descriptor.description,
            metadata=metadata,
            side_effect=side_effect,
        )

    def register_tool(
        self,
        tool_registry: ToolRegistry,
        qualified_name: str,
    ) -> ToolSpec:
        if not isinstance(tool_registry, ToolRegistry):
            raise AgentNetConfigurationError(
                "MCP tools must be registered into a ToolRegistry"
            )
        spec = self.to_tool_spec(qualified_name)
        return tool_registry.register(
            spec.name,
            _make_tool_callable(self, qualified_name),
            description=spec.description,
            metadata=spec.metadata,
            side_effect=spec.side_effect,
        )

    def register_all(self, tool_registry: ToolRegistry) -> tuple[ToolSpec, ...]:
        return tuple(
            self.register_tool(tool_registry, qualified_name)
            for qualified_name in self.registry.qualified_tool_names
        )

    async def aexecute(
        self,
        qualified_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        agent: Any | None = None,
        context: Any | None = None,
    ) -> Any:
        if agent is not None and hasattr(agent, "require_tool"):
            agent.require_tool(qualified_name)

        descriptor = self.registry.get_tool(qualified_name)
        server_name, tool_name = _split_qualified_name(qualified_name)
        descriptor_hash = hash_mcp_descriptor(descriptor)
        approved = False
        if self.approvals is not None:
            if self.require_approval:
                self.approvals.require_approval(qualified_name, descriptor)
                approved = True
            else:
                approved = self.approvals.is_approved(qualified_name, descriptor)

        arguments_copy = dict(arguments or {})
        _record_mcp_event(
            context,
            agent=agent,
            approved=approved,
            descriptor_hash=descriptor_hash,
            event_type="mcp.tool.called",
            server=server_name,
            tool=tool_name,
        )
        try:
            result = await self._call_client(server_name, tool_name, arguments_copy)
        except Exception as exc:
            _record_mcp_event(
                context,
                agent=agent,
                approved=approved,
                descriptor_hash=descriptor_hash,
                event_type="mcp.tool.failed",
                server=server_name,
                tool=tool_name,
                error=exc,
            )
            raise

        _record_mcp_event(
            context,
            agent=agent,
            approved=approved,
            descriptor_hash=descriptor_hash,
            event_type="mcp.tool.completed",
            server=server_name,
            tool=tool_name,
        )
        return result

    def execute(
        self,
        qualified_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        agent: Any | None = None,
        context: Any | None = None,
    ) -> Any:
        async def _run() -> Any:
            return await self.aexecute(
                qualified_name,
                arguments,
                agent=agent,
                context=context,
            )

        return anyio.run(_run)

    async def _call_client(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        if hasattr(self.client, "acall_tool"):
            result = self.client.acall_tool(server_name, tool_name, arguments)
        elif hasattr(self.client, "call_tool"):
            result = self.client.call_tool(server_name, tool_name, arguments)
        else:
            raise AgentNetConfigurationError(
                "MCP client must define acall_tool or call_tool"
            )

        if inspect.isawaitable(result):
            return await result
        return result


def _make_tool_callable(
    adapter: MCPToolAdapter,
    qualified_name: str,
) -> Callable[..., Any]:
    async def _implementation(
        *,
        _agentnet_agent: Any | None = None,
        _agentnet_context: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        return await adapter.aexecute(
            qualified_name,
            kwargs,
            agent=_agentnet_agent,
            context=_agentnet_context,
        )

    implementation: Any = _implementation
    implementation._agentnet_accepts_context = True
    return _implementation


def _record_mcp_event(
    context: Any | None,
    *,
    agent: Any | None,
    approved: bool,
    descriptor_hash: str,
    event_type: str,
    server: str,
    tool: str,
    error: Exception | None = None,
) -> None:
    if context is None or not hasattr(context, "metadata"):
        return

    mcp_events = context.metadata.setdefault("mcp_events", [])
    if not isinstance(mcp_events, list):
        return

    event: dict[str, Any] = {
        "agent": getattr(agent, "name", None),
        "approved": approved,
        "descriptor_hash": descriptor_hash,
        "server": server,
        "tool": tool,
        "type": event_type,
    }
    if error is not None:
        event["error_type"] = type(error).__name__
    mcp_events.append(event)


def _split_qualified_name(qualified_name: str) -> tuple[str, str]:
    server_name, separator, tool_name = qualified_name.partition(".")
    if not separator or not server_name or not tool_name:
        raise AgentNetConfigurationError(
            "MCP tool name must be qualified as 'server.tool'"
        )
    return server_name, tool_name
