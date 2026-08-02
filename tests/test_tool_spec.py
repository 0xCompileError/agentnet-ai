import pytest

import agentnet as an
from agentnet.tools import ToolSpec


def test_tool_spec_stores_descriptor_fields() -> None:
    metadata = {"category": "search", "requires_confirmation": False}
    spec = ToolSpec(
        name="search_docs",
        description="Search internal docs.",
        metadata=metadata,
        side_effect=False,
    )
    metadata["category"] = "changed"

    assert spec.name == "search_docs"
    assert spec.description == "Search internal docs."
    assert spec.metadata == {"category": "search", "requires_confirmation": False}
    assert spec.side_effect is False


def test_tool_spec_rejects_empty_name() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="name"):
        ToolSpec("")


def test_tool_spec_round_trips_to_dict_without_implementation() -> None:
    spec = ToolSpec(
        name="create_ticket",
        description="Create a support ticket.",
        side_effect=True,
    )

    serialized = spec.to_dict()
    restored = ToolSpec.from_dict(serialized)

    assert restored == spec
    assert serialized == {
        "description": "Create a support ticket.",
        "metadata": {},
        "name": "create_ticket",
        "side_effect": True,
    }


def test_tool_spec_rejects_secret_metadata_keys() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="metadata"):
        ToolSpec("search_docs", metadata={"api_key": "secret"})


def test_tool_spec_validates_input_and_output_values() -> None:
    spec = ToolSpec(
        "search_docs",
        input_schema=an.Schema({"query": str}),
        output_schema=list[str],
    )

    assert spec.validate_input({"query": "agentnet"}) == {"query": "agentnet"}
    assert spec.validate_output(["doc-1"]) == ["doc-1"]


def test_tool_spec_rejects_invalid_input_and_output_values() -> None:
    spec = ToolSpec(
        "search_docs",
        input_schema=an.Schema({"query": str}),
        output_schema=list[str],
    )

    with pytest.raises(an.AgentNetValidationError, match="tool input"):
        spec.validate_input({"query": 3})

    with pytest.raises(an.AgentNetValidationError, match="tool output"):
        spec.validate_output(["doc-1", 3])


def test_tool_spec_is_exported_from_package_root() -> None:
    assert an.ToolSpec is ToolSpec
