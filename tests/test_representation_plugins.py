import pytest

import agentnet as an


class UpperTextRepresentation(an.Representation):
    def __init__(self) -> None:
        super().__init__("upper_text", schema=str, media_type="text/plain")

    def validate(self, value: object, *, label: str = "output") -> str:
        if not isinstance(value, str):
            raise an.AgentNetValidationError(f"{label} must be text")
        if not value.isupper():
            raise an.AgentNetValidationError(f"{label} must be uppercase text")
        return value


def test_representation_plugin_registry_creates_registered_representations() -> None:
    registry = an.RepresentationPluginRegistry()
    registry.register("upper_text", UpperTextRepresentation)

    representation = registry.create("upper_text")

    assert isinstance(representation, UpperTextRepresentation)
    assert representation.validate("READY") == "READY"


def test_representation_plugin_registry_rejects_unknown_plugin() -> None:
    registry = an.RepresentationPluginRegistry()

    with pytest.raises(an.AgentNetConfigurationError, match="Unknown"):
        registry.create("missing")


def test_representation_plugin_registry_rejects_non_representation_results() -> None:
    registry = an.RepresentationPluginRegistry()
    registry.register("bad", lambda: object())

    with pytest.raises(an.AgentNetConfigurationError, match="Representation"):
        registry.create("bad")


def test_representation_plugin_registry_rejects_duplicate_names() -> None:
    registry = an.RepresentationPluginRegistry()
    registry.register("upper_text", UpperTextRepresentation)

    with pytest.raises(an.AgentNetConfigurationError, match="already registered"):
        registry.register("upper_text", UpperTextRepresentation)


def test_representation_plugin_registry_serializes_plugin_names_only() -> None:
    registry = an.RepresentationPluginRegistry()
    registry.register("upper_text", UpperTextRepresentation)

    assert registry.to_dict() == {"plugins": ["upper_text"]}


def test_representation_plugin_registry_is_exported_from_package_root() -> None:
    assert an.RepresentationPluginRegistry is not None
