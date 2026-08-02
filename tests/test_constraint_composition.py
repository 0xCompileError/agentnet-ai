import pytest

import agentnet as an
from agentnet.constraints import AndConstraint, Constraint, OrConstraint


class GreaterThanConstraint(Constraint):
    def __init__(self, name: str, threshold: int) -> None:
        super().__init__(name)
        self.threshold = threshold

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, int) and candidate > self.threshold


class EvenConstraint(Constraint):
    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, int) and candidate % 2 == 0


def test_and_constraint_requires_all_children_to_pass() -> None:
    constraint = AndConstraint(
        GreaterThanConstraint("gt_10", 10),
        EvenConstraint("even"),
        name="gt_10_and_even",
    )

    passed = constraint.evaluate(12)
    failed = constraint.evaluate(11)

    assert passed.passed is True
    assert failed.passed is False
    assert failed.blocks_candidate is True
    assert [result["constraint"] for result in failed.metadata["results"]] == [
        "gt_10",
        "even",
    ]


def test_or_constraint_requires_any_child_to_pass() -> None:
    constraint = OrConstraint(
        GreaterThanConstraint("gt_10", 10),
        EvenConstraint("even"),
        name="gt_10_or_even",
    )

    assert constraint.evaluate(8).passed is True
    assert constraint.evaluate(9).passed is False


def test_not_constraint_inverts_child_result() -> None:
    constraint = ~EvenConstraint("even")

    assert constraint.evaluate(3).passed is True
    assert constraint.evaluate(4).passed is False


def test_constraints_compose_with_boolean_operators() -> None:
    constraint = (GreaterThanConstraint("gt_10", 10) & EvenConstraint("even")) | (
        ~GreaterThanConstraint("gt_0", 0)
    )

    assert constraint.evaluate(12).passed is True
    assert constraint.evaluate(-1).passed is True
    assert constraint.evaluate(11).passed is False


def test_composition_rejects_empty_child_sets() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="at least one"):
        AndConstraint()

    with pytest.raises(an.AgentNetConfigurationError, match="at least one"):
        OrConstraint()


def test_composite_constraints_are_exported_from_package_root() -> None:
    assert an.AndConstraint is AndConstraint
    assert an.OrConstraint is OrConstraint
    assert hasattr(an, "NotConstraint")
