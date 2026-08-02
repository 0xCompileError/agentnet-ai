import pytest

import agentnet as an


def create_ticket(title: str) -> dict[str, str]:
    return {"id": "TICKET-1", "title": title}


@pytest.mark.anyio
async def test_tool_runtime_combines_registry_validation_permission_and_tracing() -> None:
    registry = an.ToolRegistry()
    registry.register(
        "create_ticket",
        create_ticket,
        description="Create a support ticket.",
        input_schema=an.Schema({"title": str}),
        metadata={"category": "support"},
        output_schema=an.Schema({"id": str, "title": str}),
        side_effect=True,
    )
    agent = an.ReActAgent("operator", tools=["create_ticket"])
    context = an.RunContext(run_id="run-1")

    result = await registry.aexecute(
        "create_ticket",
        {"title": "Investigate latency"},
        agent=agent,
        context=context,
    )

    assert result == {"id": "TICKET-1", "title": "Investigate latency"}
    assert registry.to_dict() == {
        "tools": [
            {
                "description": "Create a support ticket.",
                "metadata": {"category": "support"},
                "name": "create_ticket",
                "side_effect": True,
            }
        ]
    }
    assert context.metadata["tool_events"] == [
        {
            "agent": "operator",
            "side_effect": True,
            "tool": "create_ticket",
            "type": "tool.called",
        },
        {
            "agent": "operator",
            "side_effect": True,
            "tool": "create_ticket",
            "type": "tool.completed",
        },
    ]
