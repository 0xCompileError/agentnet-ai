import pytest

import agentnet as an


def test_interface_stores_configuration_defensively() -> None:
    metadata = {"owner": "research"}
    interface = an.Interface(
        an.Schema({"summary": str}),
        name="research_summary",
        description="Research summary contract.",
        metadata=metadata,
    )
    metadata["owner"] = "changed"

    assert interface.name == "research_summary"
    assert interface.description == "Research summary contract."
    assert interface.metadata == {"owner": "research"}


def test_interface_validates_with_configured_schema() -> None:
    interface = an.Interface(an.Schema({"summary": str}))

    assert interface.validate({"summary": "ok"}) == {"summary": "ok"}

    with pytest.raises(an.AgentNetValidationError, match="payload.summary"):
        interface.validate({"summary": 3}, label="payload")


def test_interface_validates_semantic_contract_required_fields() -> None:
    interface = an.Interface(
        semantic_contract=an.SemanticContract(
            required_fields=["summary", "risks"],
        )
    )

    value = {"summary": "ok", "risks": ["latency"]}

    assert interface.validate(value) == value

    with pytest.raises(an.AgentNetValidationError, match="payload.risks"):
        interface.validate({"summary": "ok"}, label="payload")


def test_semantic_contract_rejects_empty_required_field_names() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="required field"):
        an.SemanticContract(required_fields=[""])


def test_interface_stores_multiple_representations_defensively() -> None:
    json_representation = an.Representation("json")
    markdown_representation = an.Representation("markdown")
    representations = [json_representation, markdown_representation]

    interface = an.Interface(representations=representations)
    representations.append(an.Representation("xml"))

    assert interface.representations == (json_representation, markdown_representation)
    assert interface.representation_identifiers == ("json", "markdown")
    assert interface.supports_representation("json") is True
    assert interface.supports_representation("xml") is False
    assert interface.get_representation("markdown") is markdown_representation


def test_interface_rejects_duplicate_representation_identifiers() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="Duplicate representation"):
        an.Interface(
            representations=[
                an.Representation("json"),
                an.Representation("json"),
            ]
        )


def test_interface_validates_selected_representation_schema() -> None:
    interface = an.Interface(
        representations=[
            an.Representation("json", schema=dict[str, str]),
            an.Representation("plain_text", schema=str),
        ]
    )

    assert interface.validate({"summary": "ok"}, representation="json") == {
        "summary": "ok"
    }
    assert interface.validate("ok", representation="plain_text") == "ok"

    with pytest.raises(an.AgentNetValidationError, match="output"):
        interface.validate("not json", representation="json")


def test_interface_validates_schema_representation_and_semantics_together() -> None:
    interface = an.Interface(
        schema=an.Schema({"summary": str}),
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.KeyValueRepresentation(required_keys=["summary"])],
    )

    assert interface.validate(
        {"summary": "ok"},
        representation="key_value",
    ) == {"summary": "ok"}

    with pytest.raises(an.AgentNetValidationError, match="candidate.summary"):
        interface.validate(
            {"summary": 3},
            label="candidate",
            representation="key_value",
        )


def test_interface_without_schema_allows_any_value() -> None:
    interface = an.Interface()

    assert interface.validate("free form") == "free form"


def test_interface_rejects_empty_name() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="name"):
        an.Interface(name="")


def test_interface_is_exported_from_package_root() -> None:
    assert an.Interface is not None
    assert an.SemanticContract is not None
