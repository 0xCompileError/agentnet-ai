import pytest

import agentnet as an


def test_json_schema_representation_validates_object_schema() -> None:
    representation = an.JSONSchemaRepresentation(
        {
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {"type": "string"},
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
    )

    value = {"summary": "ok", "risks": ["latency"]}

    assert representation.identifier == "json_schema"
    assert representation.media_type == "application/schema+json"
    assert representation.validate(value, label="payload") == value


def test_json_schema_representation_rejects_invalid_values() -> None:
    representation = an.JSONSchemaRepresentation(
        {
            "type": "object",
            "required": ["summary"],
            "properties": {"summary": {"type": "string"}},
        }
    )

    with pytest.raises(an.AgentNetValidationError, match="payload.summary"):
        representation.validate({}, label="payload")

    with pytest.raises(an.AgentNetValidationError, match="payload.summary"):
        representation.validate({"summary": 3}, label="payload")


def test_json_schema_representation_stores_schema_defensively() -> None:
    json_schema = {"type": "object", "required": ["summary"]}
    representation = an.JSONSchemaRepresentation(json_schema)
    json_schema["required"] = []

    assert representation.json_schema == {"type": "object", "required": ["summary"]}


def test_json_schema_representation_is_exported_from_package_root() -> None:
    assert an.JSONSchemaRepresentation is not None
