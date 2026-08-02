import pytest

import agentnet as an


def test_mcp_server_stores_configuration_defensively() -> None:
    command = ["npx", "-y", "@modelcontextprotocol/server-github"]
    metadata = {"owner": "platform"}
    server = an.MCPServer(
        name="github",
        command=command,
        metadata=metadata,
        env={"GITHUB_TOKEN": "secret"},
    )
    command.append("--changed")
    metadata["owner"] = "changed"

    assert server.name == "github"
    assert server.command == ("npx", "-y", "@modelcontextprotocol/server-github")
    assert server.enabled is False
    assert server.metadata == {"owner": "platform"}


def test_mcp_server_qualifies_registered_tool_names() -> None:
    server = an.MCPServer(
        name="github",
        command=["mcp-github"],
        tools=[
            an.MCPToolDescriptor("search_repos"),
            an.MCPToolDescriptor("read_issue"),
        ],
    )

    assert server.tool_names == ("search_repos", "read_issue")
    assert server.qualified_tool_names == ("github.search_repos", "github.read_issue")
    assert server.get_tool("search_repos").name == "search_repos"


def test_mcp_server_serializes_descriptors_without_env_secrets() -> None:
    server = an.MCPServer(
        name="github",
        command=["mcp-github"],
        enabled=True,
        env={"GITHUB_TOKEN": "secret"},
        tools=[
            an.MCPToolDescriptor(
                "search_repos",
                description="Search repositories.",
                metadata={"side_effect": False},
            )
        ],
    )

    assert server.to_dict() == {
        "command": ["mcp-github"],
        "enabled": True,
        "metadata": {},
        "name": "github",
        "tools": [
            {
                "description": "Search repositories.",
                "metadata": {"side_effect": False},
                "name": "search_repos",
            }
        ],
    }


def test_mcp_server_rejects_invalid_configuration() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="name"):
        an.MCPServer(name="", command=["mcp"])

    with pytest.raises(an.AgentNetConfigurationError, match="command"):
        an.MCPServer(name="github", command=[])

    with pytest.raises(an.AgentNetConfigurationError, match="command"):
        an.MCPServer(name="github", command="mcp")

    with pytest.raises(an.AgentNetConfigurationError, match="Duplicate"):
        an.MCPServer(
            name="github",
            command=["mcp"],
            tools=[
                an.MCPToolDescriptor("search_repos"),
                an.MCPToolDescriptor("search_repos"),
            ],
        )


def test_mcp_tool_descriptor_rejects_invalid_configuration() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="name"):
        an.MCPToolDescriptor("")


def test_mcp_server_abstraction_is_exported_from_package_root() -> None:
    assert an.MCPServer is not None
    assert an.MCPToolDescriptor is not None
