import json

import pytest

import agentnet as an


def test_interface_serializes_to_descriptor_dict() -> None:
    interface = an.Interface(
        an.Schema({"summary": str}),
        name="research_summary",
        description="Research handoff.",
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[
            an.Representation(
                "json",
                schema=dict[str, str],
                media_type="application/json",
                description="JSON payload.",
            )
        ],
        metadata={"owner": "research"},
    )

    serialized = interface.to_dict()

    json.dumps(serialized, sort_keys=True)
    assert serialized == {
        "description": "Research handoff.",
        "metadata": {"owner": "research"},
        "name": "research_summary",
        "representations": [
            {
                "description": "JSON payload.",
                "identifier": "json",
                "media_type": "application/json",
                "metadata": {},
                "schema": "dict[str, str]",
            }
        ],
        "schema": "Schema(fields={'summary': <class 'str'>})",
        "semantic_contract": {
            "metadata": {},
            "required_fields": ["summary"],
        },
        "type": "Interface",
        "version": "1",
    }
    assert an.InterfaceDescriptor.from_dict(serialized).to_dict() == serialized


def test_representation_descriptor_round_trips() -> None:
    representation = an.Representation(
        "markdown",
        media_type="text/markdown",
        metadata={"style": "bullets"},
    )

    serialized = representation.to_dict()

    assert an.RepresentationDescriptor.from_dict(serialized).to_dict() == serialized


def test_interface_descriptor_round_trip_with_multiple_representations() -> None:
    descriptor = an.InterfaceDescriptor(
        name="handoff",
        semantic_contract=an.SemanticContractDescriptor(
            required_fields=("summary", "risks"),
            metadata={"strict": True},
        ),
        representations=(
            an.RepresentationDescriptor(
                identifier="json",
                media_type="application/json",
                metadata={"quality": "structured"},
            ),
            an.RepresentationDescriptor(
                identifier="markdown",
                media_type="text/markdown",
                metadata={"quality": "readable"},
            ),
        ),
        metadata={"owner": "research"},
        version="2",
    )

    serialized = descriptor.to_dict()

    assert an.InterfaceDescriptor.from_dict(serialized).to_dict() == serialized
    assert json.loads(json.dumps(serialized)) == serialized


def test_interface_serialization_preserves_explicit_version() -> None:
    interface = an.Interface(name="research_summary", version="2026-07")

    serialized = interface.to_dict()

    assert interface.version == "2026-07"
    assert serialized["version"] == "2026-07"
    assert an.InterfaceDescriptor.from_dict(serialized).version == "2026-07"


def test_interface_descriptor_defaults_missing_version() -> None:
    descriptor = an.InterfaceDescriptor.from_dict(
        {
            "description": None,
            "metadata": {},
            "name": "legacy",
            "representations": [],
            "schema": None,
            "semantic_contract": None,
            "type": "Interface",
        }
    )

    assert descriptor.version == "1"


def test_interface_rejects_empty_version() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="version"):
        an.Interface(version="")
