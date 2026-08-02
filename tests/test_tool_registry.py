from typing import Any

import pytest

import agentnet as an
from agentnet.tools import ToolRegistry


def search_docs(query: str) -> str:
    return f"found {query}"


def test_tool_registry_registers_specs_and_implementations() -> None:
    registry = ToolRegistry()

    spec = registry.register(
        "search_docs",
        search_docs,
        description="Search internal docs.",
        metadata={"category": "search"},
        side_effect=False,
    )

    assert spec == an.ToolSpec(
        name="search_docs",
        description="Search internal docs.",
        metadata={"category": "search"},
        side_effect=False,
    )
    assert registry.get_spec("search_docs") == spec
    assert registry.get("search_docs") is search_docs
    assert registry.names == ("search_docs",)


def test_tool_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register("search_docs", search_docs)

    with pytest.raises(an.AgentNetConfigurationError, match="already registered"):
        registry.register("search_docs", search_docs)


def test_tool_registry_rejects_non_callable_implementations() -> None:
    registry = ToolRegistry()
    implementation: Any = "not callable"

    with pytest.raises(an.AgentNetConfigurationError, match="callable"):
        registry.register("search_docs", implementation)


def test_tool_registry_rejects_unknown_tools() -> None:
    registry = ToolRegistry()

    with pytest.raises(an.AgentNetConfigurationError, match="Unknown tool"):
        registry.get("missing")

    with pytest.raises(an.AgentNetConfigurationError, match="Unknown tool"):
        registry.get_spec("missing")


def test_tool_registry_serializes_specs_without_implementations() -> None:
    registry = ToolRegistry()
    registry.register("search_docs", search_docs, description="Search docs.")

    serialized = registry.to_dict()

    assert serialized == {
        "tools": [
            {
                "description": "Search docs.",
                "metadata": {},
                "name": "search_docs",
                "side_effect": False,
            }
        ]
    }
    assert "search_docs(" not in str(serialized)


def test_tool_registry_is_exported_from_package_root() -> None:
    assert an.ToolRegistry is ToolRegistry
