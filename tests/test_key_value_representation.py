import pytest

import agentnet as an


def test_key_value_representation_accepts_string_keyed_mappings() -> None:
    representation = an.KeyValueRepresentation()

    value = {"summary": "ok", "count": 2}

    assert representation.identifier == "key_value"
    assert representation.media_type == "application/vnd.agentnet.key-value"
    assert representation.validate(value) == value


def test_key_value_representation_enforces_required_keys() -> None:
    representation = an.KeyValueRepresentation(required_keys=["summary", "risks"])

    assert representation.validate({"summary": "ok", "risks": []}) == {
        "summary": "ok",
        "risks": [],
    }

    with pytest.raises(an.AgentNetValidationError, match="payload.risks"):
        representation.validate({"summary": "ok"}, label="payload")


def test_key_value_representation_can_restrict_value_types() -> None:
    representation = an.KeyValueRepresentation(value_types=(str, int))

    assert representation.validate({"summary": "ok", "count": 2}) == {
        "summary": "ok",
        "count": 2,
    }

    with pytest.raises(an.AgentNetValidationError, match="payload.enabled"):
        representation.validate({"enabled": True}, label="payload")


def test_key_value_representation_rejects_invalid_payloads() -> None:
    representation = an.KeyValueRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate("summary=ok", label="payload")

    with pytest.raises(an.AgentNetValidationError, match="key"):
        representation.validate({1: "ok"}, label="payload")


def test_key_value_representation_rejects_invalid_configuration() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="required key"):
        an.KeyValueRepresentation(required_keys=[""])

    with pytest.raises(an.AgentNetConfigurationError, match="value_types"):
        an.KeyValueRepresentation(value_types=())


def test_key_value_representation_is_exported_from_package_root() -> None:
    assert an.KeyValueRepresentation is not None
