# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


def main() -> None:
    fake = an.FakeMCPServer(name="docs")
    fake.register_tool(
        "search",
        lambda query: {"matches": [f"guide:{query}"]},
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"matches": list}),
    )
    registry = fake.to_registry(allow_tools=["search"])
    approvals = an.MCPApprovalStore()
    approvals.approve("docs.search", registry.get_tool("docs.search"), approved_by="platform")

    tools = an.ToolRegistry()
    adapter = an.MCPToolAdapter(
        registry,
        fake,
        approvals=approvals,
        require_approval=True,
    )
    adapter.register_all(tools)
    agent = an.ReActAgent(
        "researcher",
        tools=["docs.search"],
        llms=[an.FakeLLM(["Use the docs result."], name="strong")],
    )
    context = an.RunContext(run_id="example-mcp")

    result = tools.execute(
        "docs.search",
        {"query": "serialization"},
        agent=agent,
        context=context,
    )
    emit(
        "mcp",
        {
            "mcp_events": len(context.metadata["mcp_events"]),
            "result": result,
            "tool": "docs.search",
        },
    )


if __name__ == "__main__":
    main()
