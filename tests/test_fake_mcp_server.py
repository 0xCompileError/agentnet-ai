import pytest

import agentnet as an
from agentnet.mcp import FakeMCPServer


@pytest.mark.anyio
async def test_fake_mcp_server_exposes_descriptors_and_executes_tools() -> None:
    fake = an.FakeMCPServer(
        name="github",
        command=["fake-mcp-github"],
        metadata={"owner": "platform"},
    )
    spec = fake.register_tool(
        "search_repos",
        lambda query: {"items": [query]},
        description="Search repositories.",
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"items": list}),
    )

    result = await fake.acall_tool("github", "search_repos", {"query": "agentnet"})

    assert spec.name == "search_repos"
    assert result == {"items": ["agentnet"]}
    assert fake.server.to_dict() == {
        "command": ["fake-mcp-github"],
        "enabled": True,
        "metadata": {"owner": "platform"},
        "name": "github",
        "tools": [
            {
                "description": "Search repositories.",
                "metadata": {},
                "name": "search_repos",
            }
        ],
    }
    assert fake.call_log == (
        {
            "arguments": {"query": "agentnet"},
            "server": "github",
            "tool": "search_repos",
        },
    )


@pytest.mark.anyio
async def test_fake_mcp_server_validates_inputs_outputs_and_server_name() -> None:
    fake = an.FakeMCPServer(name="github")
    fake.register_tool(
        "search_repos",
        lambda query: {"items": query},
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"items": list}),
    )

    with pytest.raises(an.AgentNetValidationError, match="tool input"):
        await fake.acall_tool("github", "search_repos", {"query": 1})

    with pytest.raises(an.AgentNetValidationError, match="tool output"):
        await fake.acall_tool("github", "search_repos", {"query": "agentnet"})

    with pytest.raises(an.AgentNetValidationError, match="server"):
        await fake.acall_tool("linear", "search_repos", {"query": "agentnet"})


def test_fake_mcp_server_builds_registry_with_allowlists() -> None:
    fake = an.FakeMCPServer(name="github")
    fake.register_tool("search_repos", lambda query: query)
    fake.register_tool("read_issue", lambda issue_id: issue_id)

    registry = fake.to_registry(allow_tools=["search_repos"])

    assert registry.qualified_tool_names == ("github.search_repos",)


def test_fake_mcp_server_is_exported_from_package_root_and_subpackage() -> None:
    assert an.FakeMCPServer is FakeMCPServer
