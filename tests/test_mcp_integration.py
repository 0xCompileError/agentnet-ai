import pytest

import agentnet as an


@pytest.mark.anyio
async def test_mcp_integration_runs_approved_allowlisted_fake_tool_through_tool_registry() -> None:
    fake = an.FakeMCPServer(name="github", env={"GITHUB_TOKEN": "secret"})
    fake.register_tool(
        "search_repos",
        lambda query: {"items": [f"repo:{query}"]},
        description="Search repositories.",
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"items": list}),
    )
    fake.register_tool("delete_repo", lambda repo: {"deleted": repo}, side_effect=True)
    mcp_registry = fake.to_registry(allow_tools=["search_repos"])
    approvals = an.MCPApprovalStore()
    approvals.approve(
        "github.search_repos",
        mcp_registry.get_tool("github.search_repos"),
        approved_by="platform",
    )
    adapter = an.MCPToolAdapter(
        mcp_registry,
        fake,
        approvals=approvals,
        require_approval=True,
    )
    tool_registry = an.ToolRegistry()

    adapter.register_all(tool_registry)
    agent = an.ReActAgent("researcher", tools=["github.search_repos"])
    context = an.RunContext("mcp-integration")
    result = await tool_registry.aexecute(
        "github.search_repos",
        {"query": "agentnet"},
        agent=agent,
        context=context,
    )

    assert result == {"items": ["repo:agentnet"]}
    assert tool_registry.names == ("github.search_repos",)
    assert "github.delete_repo" not in tool_registry.names
    assert context.metadata["tool_events"] == [
        {
            "agent": "researcher",
            "side_effect": False,
            "tool": "github.search_repos",
            "type": "tool.called",
        },
        {
            "agent": "researcher",
            "side_effect": False,
            "tool": "github.search_repos",
            "type": "tool.completed",
        },
    ]
    assert context.metadata["mcp_events"][0]["type"] == "mcp.tool.called"
    assert context.metadata["mcp_events"][1]["type"] == "mcp.tool.completed"
    assert "secret" not in str(mcp_registry.to_dict())
    assert "GITHUB_TOKEN" not in str(mcp_registry.to_dict())
