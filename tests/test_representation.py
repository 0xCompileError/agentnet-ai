import pytest

import agentnet as an


def test_representation_stores_configuration_defensively() -> None:
    metadata = {"format": "structured"}
    representation = an.Representation(
        "json",
        schema=dict[str, str],
        media_type="application/json",
        description="JSON object representation.",
        metadata=metadata,
    )
    metadata["format"] = "changed"

    assert representation.identifier == "json"
    assert representation.schema == dict[str, str]
    assert representation.media_type == "application/json"
    assert representation.description == "JSON object representation."
    assert representation.metadata == {"format": "structured"}


def test_representation_validates_with_configured_schema() -> None:
    representation = an.Representation("json", schema=dict[str, str])

    assert representation.validate({"summary": "ok"}) == {"summary": "ok"}

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate("not a mapping", label="payload")


def test_representation_without_schema_allows_any_value() -> None:
    representation = an.Representation("plain_text")

    assert representation.validate("free form") == "free form"


def test_representation_rejects_empty_identifier() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="identifier"):
        an.Representation("")


def test_representation_is_exported_from_package_root() -> None:
    assert an.Representation is not None
