"""MCP descriptor approval records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.mcp._security import validate_safe_metadata
from agentnet.mcp.descriptors import hash_mcp_descriptor


@dataclass(frozen=True, slots=True, init=False)
class MCPDescriptorApproval:
    """Pinned approval for an MCP descriptor hash."""

    target: str
    descriptor_hash: str
    approved_by: str | None
    reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        target: str,
        descriptor_hash: str,
        approved_by: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not target:
            raise AgentNetConfigurationError("MCP approval target cannot be empty")
        if not descriptor_hash:
            raise AgentNetConfigurationError(
                "MCP approval descriptor_hash cannot be empty"
            )
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="MCP approval")

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "descriptor_hash", descriptor_hash)
        object.__setattr__(self, "approved_by", approved_by)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_by": self.approved_by,
            "descriptor_hash": self.descriptor_hash,
            "metadata": self.metadata.copy(),
            "reason": self.reason,
            "target": self.target,
        }


class MCPApprovalStore:
    """In-memory MCP descriptor approval store."""

    def __init__(
        self,
        approvals: Iterable[MCPDescriptorApproval] | None = None,
    ) -> None:
        self._approvals: dict[str, MCPDescriptorApproval] = {}
        for approval in approvals or ():
            self.register(approval)

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self._approvals)

    def register(self, approval: MCPDescriptorApproval) -> MCPDescriptorApproval:
        if not isinstance(approval, MCPDescriptorApproval):
            raise AgentNetConfigurationError(
                "MCP approvals must be MCPDescriptorApproval instances"
            )
        self._approvals[approval.target] = approval
        return approval

    def approve(
        self,
        target: str,
        descriptor: Any,
        *,
        approved_by: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MCPDescriptorApproval:
        approval = MCPDescriptorApproval(
            target=target,
            descriptor_hash=hash_mcp_descriptor(descriptor),
            approved_by=approved_by,
            reason=reason,
            metadata=metadata,
        )
        return self.register(approval)

    def get(self, target: str) -> MCPDescriptorApproval:
        try:
            return self._approvals[target]
        except KeyError as exc:
            raise AgentNetValidationError(
                f"MCP descriptor {target!r} is not approved"
            ) from exc

    def require_approval(self, target: str, descriptor: Any) -> MCPDescriptorApproval:
        approval = self.get(target)
        descriptor_hash = hash_mcp_descriptor(descriptor)
        if approval.descriptor_hash != descriptor_hash:
            raise AgentNetValidationError(
                f"MCP descriptor approval hash mismatch for {target!r}"
            )
        return approval

    def is_approved(self, target: str, descriptor: Any) -> bool:
        try:
            self.require_approval(target, descriptor)
        except AgentNetValidationError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "approvals": [
                approval.to_dict() for approval in self._approvals.values()
            ],
        }
