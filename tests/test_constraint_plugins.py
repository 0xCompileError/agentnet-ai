import pytest

import agentnet as an
from agentnet.constraints import Constraint, ConstraintPluginRegistry


class EqualsConstraint(Constraint):
    def __init__(self, expected: object) -> None:
        super().__init__("equals")
        self.expected = expected

    def check(self, candidate: object, context: object | None = None) -> bool:
        return candidate == self.expected


def test_constraint_plugin_registry_creates_registered_constraints() -> None:
    registry = ConstraintPluginRegistry()
    registry.register("equals", lambda expected: EqualsConstraint(expected))

    constraint = registry.create("equals", expected="value")

    assert isinstance(constraint, EqualsConstraint)
    assert constraint.evaluate("value").passed is True
    assert constraint.evaluate("other").passed is False


def test_constraint_plugin_registry_rejects_unknown_plugin() -> None:
    registry = ConstraintPluginRegistry()

    with pytest.raises(an.AgentNetConfigurationError, match="Unknown"):
        registry.create("missing")


def test_constraint_plugin_registry_rejects_non_constraint_factory_result() -> None:
    registry = ConstraintPluginRegistry()
    registry.register("bad", lambda: object())

    with pytest.raises(an.AgentNetConfigurationError, match="Constraint"):
        registry.create("bad")


def test_constraint_plugin_registry_serializes_plugin_names_only() -> None:
    registry = ConstraintPluginRegistry()
    registry.register("equals", lambda expected: EqualsConstraint(expected))

    assert registry.to_dict() == {"plugins": ["equals"]}


def test_constraint_plugin_registry_is_exported_from_package_root() -> None:
    assert an.ConstraintPluginRegistry is ConstraintPluginRegistry
