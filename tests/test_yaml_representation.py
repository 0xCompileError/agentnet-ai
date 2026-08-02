import pytest

import agentnet as an


def test_yaml_representation_accepts_yaml_text() -> None:
    representation = an.YAMLRepresentation()

    yaml = "summary: ok\nrisks:\n  - latency"

    assert representation.identifier == "yaml"
    assert representation.media_type == "application/yaml"
    assert representation.validate(yaml) == yaml


def test_yaml_representation_can_require_mapping() -> None:
    representation = an.YAMLRepresentation(require_mapping=True)

    assert representation.validate("summary: ok") == "summary: ok"

    with pytest.raises(an.AgentNetValidationError, match="mapping"):
        representation.validate("- item", label="payload")


def test_yaml_representation_rejects_non_text_values() -> None:
    representation = an.YAMLRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate({"summary": "ok"}, label="payload")


def test_yaml_representation_is_exported_from_package_root() -> None:
    assert an.YAMLRepresentation is not None
