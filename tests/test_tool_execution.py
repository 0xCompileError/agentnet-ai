import pytest

import agentnet as an


def search_docs(query: str) -> list[str]:
    return [f"doc:{query}"]


async def query_metrics(metric: str) -> dict[str, int]:
    return {metric: 3}


def invalid_output(query: str) -> str:
    return query


@pytest.mark.anyio
async def test_tool_registry_executes_registered_tool_with_validation() -> None:
    registry = an.ToolRegistry()
    registry.register(
        "search_docs",
        search_docs,
        input_schema=an.Schema({"query": str}),
        output_schema=list[str],
    )
    agent = an.ReActAgent("researcher", tools=["search_docs"])

    result = await registry.aexecute("search_docs", {"query": "agentnet"}, agent=agent)

    assert result == ["doc:agentnet"]


@pytest.mark.anyio
async def test_tool_registry_records_tool_trace_events() -> None:
    registry = an.ToolRegistry()
    registry.register(
        "search_docs",
        search_docs,
        input_schema=an.Schema({"query": str}),
        output_schema=list[str],
    )
    agent = an.ReActAgent("researcher", tools=["search_docs"])
    context = an.RunContext(run_id="run-1")

    await registry.aexecute(
        "search_docs",
        {"query": "agentnet"},
        agent=agent,
        context=context,
    )

    assert context.metadata["tool_events"] == [
        {
            "agent": "researcher",
            "side_effect": False,
            "tool": "search_docs",
            "type": "tool.called",
        },
        {
            "agent": "researcher",
            "side_effect": False,
            "tool": "search_docs",
            "type": "tool.completed",
        },
    ]


@pytest.mark.anyio
async def test_tool_registry_executes_async_tools() -> None:
    registry = an.ToolRegistry()
    registry.register(
        "query_metrics",
        query_metrics,
        input_schema=an.Schema({"metric": str}),
        output_schema=dict[str, int],
    )

    result = await registry.aexecute("query_metrics", {"metric": "latency"})

    assert result == {"latency": 3}


@pytest.mark.anyio
async def test_tool_registry_rejects_disallowed_agent_tool_execution() -> None:
    registry = an.ToolRegistry()
    registry.register("search_docs", search_docs)
    agent = an.ReActAgent("researcher", tools=["query_metrics"])

    with pytest.raises(an.AgentNetExecutionError, match="not allowed"):
        await registry.aexecute("search_docs", {"query": "agentnet"}, agent=agent)


@pytest.mark.anyio
async def test_tool_registry_validates_tool_input_before_execution() -> None:
    calls: list[str] = []

    def record(query: str) -> list[str]:
        calls.append(query)
        return [query]

    registry = an.ToolRegistry()
    registry.register("search_docs", record, input_schema=an.Schema({"query": str}))

    with pytest.raises(an.AgentNetValidationError, match="tool input"):
        await registry.aexecute("search_docs", {"query": 3})

    assert calls == []


@pytest.mark.anyio
async def test_tool_registry_validates_tool_output_after_execution() -> None:
    registry = an.ToolRegistry()
    registry.register(
        "search_docs",
        invalid_output,
        input_schema=an.Schema({"query": str}),
        output_schema=list[str],
    )

    with pytest.raises(an.AgentNetValidationError, match="tool output"):
        await registry.aexecute("search_docs", {"query": "agentnet"})


def test_tool_registry_sync_execute_wraps_async_pipeline() -> None:
    registry = an.ToolRegistry()
    registry.register("search_docs", search_docs)

    assert registry.execute("search_docs", {"query": "agentnet"}) == ["doc:agentnet"]
