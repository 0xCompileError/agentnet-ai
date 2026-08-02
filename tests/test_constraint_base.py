import pytest

import agentnet as an
from agentnet.constraints import Constraint, ConstraintKind, ConstraintResult


class NonEmptyConstraint(Constraint):
    def check(self, candidate: object, context: object | None = None) -> bool:
        return bool(candidate)


def test_constraint_evaluates_candidate_to_result() -> None:
    constraint = NonEmptyConstraint("non_empty", description="Value must be present.")

    passed = constraint.evaluate("value")
    failed = constraint.evaluate("")

    assert passed == ConstraintResult(
        constraint="non_empty",
        passed=True,
        kind=ConstraintKind.HARD,
        message=None,
        metadata={},
    )
    assert failed == ConstraintResult(
        constraint="non_empty",
        passed=False,
        kind=ConstraintKind.HARD,
        message="Constraint 'non_empty' failed",
        metadata={},
    )
    assert failed.blocks_candidate is True


def test_soft_constraint_failure_does_not_block_candidate() -> None:
    constraint = NonEmptyConstraint("preferred_non_empty", kind=ConstraintKind.SOFT)

    result = constraint.evaluate("")

    assert constraint.is_soft is True
    assert result.kind is ConstraintKind.SOFT
    assert result.passed is False
    assert result.blocks_candidate is False


def test_constraint_rejects_empty_name() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="name"):
        NonEmptyConstraint("")


def test_constraint_rejects_unknown_kind() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="kind"):
        NonEmptyConstraint("non_empty", kind="optional")


def test_constraint_result_round_trips_to_dict() -> None:
    result = ConstraintResult(
        constraint="non_empty",
        passed=False,
        kind=ConstraintKind.SOFT,
        message="missing value",
        metadata={"field": "summary"},
    )

    assert ConstraintResult.from_dict(result.to_dict()) == result
    assert result.to_dict()["kind"] == "soft"


def test_constraint_is_exported_from_package_root() -> None:
    assert an.Constraint is Constraint
    assert an.ConstraintKind is ConstraintKind
    assert an.ConstraintResult is ConstraintResult
