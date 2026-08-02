import pytest

import agentnet as an
from agentnet.constraints import CustomConstraint


def test_custom_constraint_uses_explicit_predicate() -> None:
    constraint = CustomConstraint(
        "even",
        lambda candidate, context: isinstance(candidate, int) and candidate % 2 == 0,
    )

    assert constraint.evaluate(2).passed is True
    assert constraint.evaluate(3).passed is False


def test_custom_constraint_receives_context() -> None:
    constraint = CustomConstraint(
        "matches_context",
        lambda candidate, context: candidate == context["expected"],
    )

    assert constraint.evaluate("value", context={"expected": "value"}).passed is True
    assert constraint.evaluate("other", context={"expected": "value"}).passed is False


def test_custom_constraint_descriptor_does_not_serialize_predicate() -> None:
    constraint = CustomConstraint("even", lambda candidate, context: True)

    serialized = constraint.to_dict()

    assert serialized["type"] == "CustomConstraint"
    assert serialized["parameters"] == {"custom": True}
    assert "lambda" not in str(serialized)


def test_custom_constraint_rejects_non_callable_predicate() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="callable"):
        CustomConstraint("bad", "not callable")


def test_custom_constraint_is_exported_from_package_root() -> None:
    assert an.CustomConstraint is CustomConstraint
