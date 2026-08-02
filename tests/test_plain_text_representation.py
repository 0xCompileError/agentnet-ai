import pytest

import agentnet as an


def test_plain_text_representation_accepts_text() -> None:
    representation = an.PlainTextRepresentation()

    assert representation.identifier == "plain_text"
    assert representation.media_type == "text/plain"
    assert representation.validate("hello") == "hello"


def test_plain_text_representation_can_require_non_empty_text() -> None:
    representation = an.PlainTextRepresentation(require_non_empty=True)

    assert representation.validate("hello") == "hello"

    with pytest.raises(an.AgentNetValidationError, match="non-empty"):
        representation.validate("   ", label="payload")


def test_plain_text_representation_rejects_non_text_values() -> None:
    representation = an.PlainTextRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate(["hello"], label="payload")


def test_plain_text_representation_is_exported_from_package_root() -> None:
    assert an.PlainTextRepresentation is not None
