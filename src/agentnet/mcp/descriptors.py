"""MCP descriptor hashing and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentnet.core import AgentNetConfigurationError
from agentnet.mcp._security import validate_descriptor_payload_no_secrets
from agentnet.mcp.registry import MCPRegistry
from agentnet.mcp.server import MCPServer, MCPToolDescriptor


def hash_mcp_descriptor(descriptor: Any) -> str:
    """Return a stable SHA-256 hash for an MCP descriptor."""

    descriptor_payload = _descriptor_to_dict(descriptor)
    canonical = _canonicalize(descriptor_payload)
    serialized = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_mcp_descriptor(descriptor: Any) -> Any:
    """Validate an MCP descriptor or descriptor payload."""

    if isinstance(descriptor, MCPToolDescriptor):
        validate_descriptor_payload_no_secrets(
            descriptor.to_dict(),
            label="MCP descriptor",
        )
        return descriptor

    if isinstance(descriptor, MCPServer):
        _validate_server_payload(descriptor.to_dict())
        return descriptor

    if isinstance(descriptor, MCPRegistry):
        _validate_registry_payload(descriptor.to_dict())
        for qualified_name in descriptor.qualified_tool_names:
            descriptor.get_tool(qualified_name)
        return descriptor

    if isinstance(descriptor, Mapping):
        _validate_mapping_payload(descriptor)
        return descriptor

    raise AgentNetConfigurationError(
        "MCP descriptor must be an MCP tool, server, registry, or mapping"
    )


def _descriptor_to_dict(descriptor: Any) -> Mapping[str, Any]:
    validate_mcp_descriptor(descriptor)
    if hasattr(descriptor, "to_dict"):
        value = descriptor.to_dict()
        if not isinstance(value, Mapping):
            raise AgentNetConfigurationError("MCP descriptor to_dict must return a mapping")
        return value
    if isinstance(descriptor, Mapping):
        return descriptor
    raise AgentNetConfigurationError("MCP descriptor must be serializable")


def _validate_registry_payload(payload: Mapping[str, Any]) -> None:
    validate_descriptor_payload_no_secrets(payload, label="MCP descriptor")
    servers = payload.get("servers")
    if servers is None:
        raise AgentNetConfigurationError("MCP registry descriptor requires servers")
    if not isinstance(servers, Sequence) or isinstance(servers, str | bytes | bytearray):
        raise AgentNetConfigurationError("MCP registry descriptor servers must be a sequence")
    for server in servers:
        if not isinstance(server, Mapping):
            raise AgentNetConfigurationError("MCP registry descriptor servers must be mappings")
        _validate_server_payload(server)


def _validate_mapping_payload(payload: Mapping[str, Any]) -> None:
    if "servers" in payload:
        _validate_registry_payload(payload)
        return
    if "command" in payload:
        _validate_server_payload(payload)
        return
    _validate_tool_payload(payload)


def _validate_server_payload(payload: Mapping[str, Any]) -> None:
    validate_descriptor_payload_no_secrets(payload, label="MCP descriptor")
    if not payload.get("name"):
        raise AgentNetConfigurationError("MCP server descriptor name cannot be empty")

    command = payload.get("command")
    if isinstance(command, str) or not isinstance(command, Sequence):
        raise AgentNetConfigurationError("MCP server descriptor command must be a sequence")
    if not command:
        raise AgentNetConfigurationError("MCP server descriptor command cannot be empty")
    if any(not str(part) for part in command):
        raise AgentNetConfigurationError(
            "MCP server descriptor command parts cannot be empty"
        )

    tools = payload.get("tools", ())
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes | bytearray):
        raise AgentNetConfigurationError("MCP server descriptor tools must be a sequence")
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise AgentNetConfigurationError("MCP tool descriptor must be a mapping")
        _validate_tool_payload(tool)
        tool_name = str(tool["name"])
        if tool_name in names:
            raise AgentNetConfigurationError(
                f"Duplicate MCP tool descriptor {tool_name!r}"
            )
        names.add(tool_name)


def _validate_tool_payload(payload: Mapping[str, Any]) -> None:
    validate_descriptor_payload_no_secrets(payload, label="MCP descriptor")
    if not payload.get("name"):
        raise AgentNetConfigurationError("MCP tool descriptor name cannot be empty")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonicalize(nested) for nested in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)
