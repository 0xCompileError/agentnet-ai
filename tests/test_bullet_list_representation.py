import pytest

import agentnet as an


def test_bullet_list_representation_accepts_string_items() -> None:
    representation = an.BulletListRepresentation()

    value = ["fast", "cheap"]

    assert representation.identifier == "bullet_list"
    assert representation.media_type == "application/vnd.agentnet.bullet-list"
    assert representation.validate(value) == value


def test_bullet_list_representation_enforces_min_items() -> None:
    representation = an.BulletListRepresentation(min_items=2)

    assert representation.validate(["one", "two"]) == ["one", "two"]

    with pytest.raises(an.AgentNetValidationError, match="at least 2"):
        representation.validate(["one"], label="payload")


def test_bullet_list_representation_rejects_invalid_items() -> None:
    representation = an.BulletListRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate("not a list", label="payload")

    with pytest.raises(an.AgentNetValidationError, match="payload\\[1\\]"):
        representation.validate(["ok", 3], label="payload")


def test_bullet_list_representation_is_exported_from_package_root() -> None:
    assert an.BulletListRepresentation is not None
