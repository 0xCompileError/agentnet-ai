import pytest

import agentnet as an
from agentnet.mcp import MCPToolAdapter


def test_mcp_tool_adapter_registers_tool_specs_without_implementations() -> None:
    fake = an.FakeMCPServer(name="github")
    fake.register_tool(
        "search_repos",
        lambda query: {"items": [query]},
        description="Search repositories.",
        metadata={"category": "search"},
    )
    registry = fake.to_registry()
    adapter = an.MCPToolAdapter(registry, fake)
    tools = an.ToolRegistry()

    specs = adapter.register_all(tools)

    assert len(specs) == 1
    assert specs[0].name == "github.search_repos"
    assert specs[0].description == "Search repositories."
    assert specs[0].metadata == {
        "category": "search",
        "mcp": True,
        "mcp_descriptor_hash": an.hash_mcp_descriptor(
            registry.get_tool("github.search_repos")
        ),
        "mcp_server": "github",
        "mcp_tool": "search_repos",
    }
    serialized = tools.to_dict()
    assert "lambda" not in str(serialized)
    assert "GITHUB_TOKEN" not in str(serialized)


@pytest.mark.anyio
async def test_mcp_tool_adapter_executes_approved_tools_and_records_trace_events() -> None:
    fake = an.FakeMCPServer(name="github")
    fake.register_tool(
        "search_repos",
        lambda query: {"items": [query]},
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"items": list}),
    )
    registry = fake.to_registry(allow_tools=["search_repos"])
    approvals = an.MCPApprovalStore()
    approvals.approve("github.search_repos", registry.get_tool("github.search_repos"))
    adapter = an.MCPToolAdapter(
        registry,
        fake,
        approvals=approvals,
        require_approval=True,
    )
    tools = an.ToolRegistry()
    adapter.register_all(tools)
    agent = an.ReActAgent("researcher", tools=["github.search_repos"])
    context = an.RunContext("mcp-run")

    result = await tools.aexecute(
        "github.search_repos",
        {"query": "agentnet"},
        agent=agent,
        context=context,
    )

    descriptor_hash = an.hash_mcp_descriptor(registry.get_tool("github.search_repos"))
    assert result == {"items": ["agentnet"]}
    assert fake.call_log == (
        {
            "arguments": {"query": "agentnet"},
            "server": "github",
            "tool": "search_repos",
        },
    )
    assert context.metadata["mcp_events"] == [
        {
            "agent": "researcher",
            "approved": True,
            "descriptor_hash": descriptor_hash,
            "server": "github",
            "tool": "search_repos",
            "type": "mcp.tool.called",
        },
        {
            "agent": "researcher",
            "approved": True,
            "descriptor_hash": descriptor_hash,
            "server": "github",
            "tool": "search_repos",
            "type": "mcp.tool.completed",
        },
    ]


@pytest.mark.anyio
async def test_mcp_tool_adapter_rejects_unapproved_or_disallowed_tools() -> None:
    fake = an.FakeMCPServer(name="github")
    fake.register_tool("search_repos", lambda query: query)
    fake.register_tool("read_issue", lambda issue_id: issue_id)
    registry = fake.to_registry(allow_tools=["search_repos"])
    adapter = an.MCPToolAdapter(
        registry,
        fake,
        approvals=an.MCPApprovalStore(),
        require_approval=True,
    )

    with pytest.raises(an.AgentNetValidationError, match="not approved"):
        await adapter.aexecute("github.search_repos", {"query": "agentnet"})

    with pytest.raises(an.AgentNetValidationError, match="not allowed"):
        await adapter.aexecute("github.read_issue", {"issue_id": "1"})


def test_mcp_tool_adapter_is_exported_from_package_root_and_subpackage() -> None:
    assert an.MCPToolAdapter is MCPToolAdapter
