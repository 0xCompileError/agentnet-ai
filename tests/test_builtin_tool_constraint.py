import agentnet as an
from agentnet.constraints import ToolConstraint


def tool() -> str:
    return "ok"


def test_tool_constraint_accepts_allowed_tool_string_and_spec() -> None:
    constraint = ToolConstraint(allowed=["search_docs", "summarize"])

    assert constraint.evaluate("search_docs").passed is True
    assert constraint.evaluate(an.ToolSpec("summarize")).passed is True
    assert constraint.evaluate("delete_records").passed is False


def test_tool_constraint_requires_all_registry_tools_to_be_allowed() -> None:
    registry = an.ToolRegistry()
    registry.register("search_docs", tool)
    registry.register("summarize", tool)
    constraint = ToolConstraint(allowed=["search_docs", "summarize"])

    assert constraint.evaluate(registry).passed is True

    registry.register("delete_records", tool)
    assert constraint.evaluate(registry).passed is False


def test_tool_constraint_reads_react_agent_tools() -> None:
    constraint = ToolConstraint(allowed=["search_docs"])

    assert constraint.evaluate(an.ReActAgent("agent", tools=["search_docs"])).passed is True
    assert constraint.evaluate(an.ReActAgent("agent", tools=["delete_records"])).passed is False


def test_tool_constraint_is_exported_from_package_root() -> None:
    assert an.ToolConstraint is ToolConstraint
