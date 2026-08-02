import pytest

import agentnet as an


def test_representation_translator_applies_explicit_transform() -> None:
    translator = an.RepresentationTranslator(
        "markdown",
        "json",
        lambda value: {"text": value},
        metadata={"quality": "lossless"},
    )

    assert translator.source == "markdown"
    assert translator.target == "json"
    assert translator.metadata == {"quality": "lossless"}
    assert translator.translate("hello") == {"text": "hello"}


def test_representation_translator_rejects_invalid_configuration() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="source"):
        an.RepresentationTranslator("", "json", lambda value: value)

    with pytest.raises(an.AgentNetConfigurationError, match="target"):
        an.RepresentationTranslator("json", "", lambda value: value)

    with pytest.raises(an.AgentNetConfigurationError, match="callable"):
        an.RepresentationTranslator("json", "markdown", "not callable")


def test_representation_translator_registry_translates_registered_pair() -> None:
    registry = an.RepresentationTranslatorRegistry()
    registry.register(
        an.RepresentationTranslator(
            "markdown",
            "json",
            lambda value: {"text": value},
        )
    )

    assert registry.translate("markdown", "json", "hello") == {"text": "hello"}


def test_representation_translator_registry_exposes_registered_translators() -> None:
    translator = an.RepresentationTranslator("markdown", "json", lambda value: value)
    registry = an.RepresentationTranslatorRegistry([translator])

    assert registry.translators == (translator,)


def test_representation_translator_registry_replaces_existing_pair() -> None:
    registry = an.RepresentationTranslatorRegistry()
    registry.register(an.RepresentationTranslator("markdown", "json", lambda value: value))
    registry.register(
        an.RepresentationTranslator(
            "markdown",
            "json",
            lambda value: {"text": value},
        )
    )

    assert registry.translate("markdown", "json", "hello") == {"text": "hello"}
    assert len(registry.translators) == 1


def test_representation_translator_registry_rejects_unknown_pair() -> None:
    registry = an.RepresentationTranslatorRegistry()

    with pytest.raises(an.AgentNetValidationError, match="No translator"):
        registry.translate("markdown", "json", "hello")


def test_representation_translators_are_exported_from_package_root() -> None:
    assert an.RepresentationTranslator is not None
    assert an.RepresentationTranslatorRegistry is not None
