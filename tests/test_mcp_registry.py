import pytest

import agentnet as an
from agentnet.mcp import MCPRegistry


def _github_server() -> an.MCPServer:
    return an.MCPServer(
        name="github",
        command=["mcp-github"],
        env={"GITHUB_TOKEN": "secret"},
        tools=[
            an.MCPToolDescriptor("search_repos"),
            an.MCPToolDescriptor("read_issue"),
        ],
    )


def test_mcp_registry_registers_servers_and_qualified_tools() -> None:
    server = _github_server()
    registry = an.MCPRegistry()

    registered = registry.register(server)

    assert registered is server
    assert registry.names == ("github",)
    assert registry.server_names == ("github",)
    assert registry.qualified_tool_names == (
        "github.search_repos",
        "github.read_issue",
    )
    assert registry.get_server("github") is server
    assert registry.get_tool("github.search_repos").name == "search_repos"


def test_mcp_registry_constructor_registers_initial_servers() -> None:
    github = _github_server()
    linear = an.MCPServer(
        name="linear",
        command=["mcp-linear"],
        tools=[an.MCPToolDescriptor("search_issues")],
    )

    registry = an.MCPRegistry([github, linear])

    assert registry.names == ("github", "linear")
    assert registry.qualified_tool_names == (
        "github.search_repos",
        "github.read_issue",
        "linear.search_issues",
    )


def test_mcp_registry_rejects_invalid_and_duplicate_servers() -> None:
    registry = an.MCPRegistry()
    registry.register(_github_server())

    with pytest.raises(an.AgentNetConfigurationError, match="already registered"):
        registry.register(_github_server())

    with pytest.raises(an.AgentNetConfigurationError, match="MCPServer"):
        registry.register("not a server")  # type: ignore[arg-type]


def test_mcp_registry_rejects_unknown_server_lookup() -> None:
    registry = an.MCPRegistry()

    with pytest.raises(an.AgentNetConfigurationError, match="Unknown MCP server"):
        registry.get_server("missing")


def test_mcp_registry_rejects_invalid_qualified_tool_lookup() -> None:
    registry = an.MCPRegistry([_github_server()])

    with pytest.raises(an.AgentNetValidationError, match="qualified"):
        registry.get_tool("search_repos")

    with pytest.raises(an.AgentNetValidationError, match="Unknown MCP server"):
        registry.get_tool("missing.search_repos")

    with pytest.raises(an.AgentNetValidationError, match="Unknown MCP tool"):
        registry.get_tool("github.missing")


def test_mcp_registry_serializes_server_descriptors_without_env_secrets() -> None:
    registry = an.MCPRegistry([_github_server()])

    serialized = registry.to_dict()

    assert serialized == {
        "servers": [
            {
                "command": ["mcp-github"],
                "enabled": False,
                "metadata": {},
                "name": "github",
                "tools": [
                    {
                        "description": None,
                        "metadata": {},
                        "name": "search_repos",
                    },
                    {
                        "description": None,
                        "metadata": {},
                        "name": "read_issue",
                    },
                ],
            }
        ]
    }
    assert "GITHUB_TOKEN" not in str(serialized)
    assert "secret" not in str(serialized)


def test_mcp_registry_is_exported_from_package_root_and_subpackage() -> None:
    assert an.MCPRegistry is MCPRegistry


def test_mcp_registry_restricts_tools_with_allowlists() -> None:
    registry = an.MCPRegistry()
    registry.register(_github_server(), allow_tools=["search_repos"])

    assert registry.allowed_tools("github") == ("search_repos",)
    assert registry.qualified_tool_names == ("github.search_repos",)
    assert registry.is_tool_allowed("github.search_repos") is True
    assert registry.is_tool_allowed("github.read_issue") is False

    with pytest.raises(an.AgentNetValidationError, match="not allowed"):
        registry.get_tool("github.read_issue")

    assert registry.to_dict()["allowlists"] == {"github": ["search_repos"]}


def test_mcp_registry_rejects_unknown_allowlisted_tools() -> None:
    registry = an.MCPRegistry()

    with pytest.raises(an.AgentNetConfigurationError, match="Unknown MCP tool"):
        registry.register(_github_server(), allow_tools=["missing"])
