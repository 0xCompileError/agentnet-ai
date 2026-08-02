import pytest

import agentnet as an


def _descriptor() -> an.MCPToolDescriptor:
    return an.MCPToolDescriptor(
        "search_repos",
        description="Search repositories.",
        metadata={"category": "search"},
    )


def test_mcp_descriptor_hashing_is_stable_and_secret_safe() -> None:
    first = an.MCPServer(
        name="github",
        command=["mcp-github"],
        env={"GITHUB_TOKEN": "secret-one"},
        tools=[_descriptor()],
    )
    second = an.MCPServer(
        name="github",
        command=["mcp-github"],
        env={"GITHUB_TOKEN": "secret-two"},
        tools=[_descriptor()],
    )
    changed = an.MCPServer(
        name="github",
        command=["mcp-github", "--readonly"],
        tools=[_descriptor()],
    )

    descriptor_hash = an.hash_mcp_descriptor(first)

    assert descriptor_hash == an.hash_mcp_descriptor(second)
    assert descriptor_hash != an.hash_mcp_descriptor(changed)
    assert len(descriptor_hash) == 64


def test_mcp_descriptor_validation_rejects_secret_serialization() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="secrets"):
        an.MCPToolDescriptor("search_repos", metadata={"api_token": "secret"})

    with pytest.raises(an.AgentNetConfigurationError, match="secrets"):
        an.MCPServer(
            name="github",
            command=["mcp-github"],
            metadata={"api_key": "secret"},
        )

    with pytest.raises(an.AgentNetConfigurationError, match="env"):
        an.validate_mcp_descriptor(
            {
                "name": "github",
                "command": ["mcp-github"],
                "env": {"GITHUB_TOKEN": "secret"},
            }
        )


def test_mcp_descriptor_approval_pins_hashes() -> None:
    original = _descriptor()
    changed = an.MCPToolDescriptor("search_repos", description="Changed.")
    approvals = an.MCPApprovalStore()

    approval = approvals.approve(
        "github.search_repos",
        original,
        approved_by="platform",
        reason="Read-only search tool.",
    )

    assert approval.descriptor_hash == an.hash_mcp_descriptor(original)
    assert approvals.require_approval("github.search_repos", original) == approval
    assert approvals.is_approved("github.search_repos", original) is True
    assert approvals.is_approved("github.search_repos", changed) is False

    with pytest.raises(an.AgentNetValidationError, match="hash mismatch"):
        approvals.require_approval("github.search_repos", changed)

    with pytest.raises(an.AgentNetValidationError, match="not approved"):
        approvals.require_approval("github.read_issue", original)

    assert approvals.to_dict() == {
        "approvals": [
            {
                "approved_by": "platform",
                "descriptor_hash": approval.descriptor_hash,
                "metadata": {},
                "reason": "Read-only search tool.",
                "target": "github.search_repos",
            }
        ]
    }
