from typing import Any

import pytest

import agentnet as an
from agentnet.constraints import Constraint


class ValueAtMostConstraint(Constraint):
    def __init__(
        self,
        max_value: int,
        *,
        kind: an.ConstraintKind = an.ConstraintKind.HARD,
    ) -> None:
        super().__init__(f"value_at_most_{max_value}", kind=kind)
        self.max_value = max_value

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, int) and candidate <= self.max_value


def test_validate_training_constraints_returns_results_and_records_summaries() -> None:
    metadata: dict[str, Any] = {}

    results = an.validate_training_constraints(
        3,
        [ValueAtMostConstraint(5)],
        metadata=metadata,
    )

    assert results[0].passed is True
    assert metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "value_at_most_5",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_validate_training_constraints_rejects_failed_hard_constraint() -> None:
    metadata: dict[str, Any] = {}

    with pytest.raises(an.AgentNetValidationError, match="value_at_most_5"):
        an.validate_training_constraints(
            6,
            [ValueAtMostConstraint(5)],
            metadata=metadata,
        )

    assert metadata["training_constraint_results"][0]["blocks_candidate"] is True


def test_validate_training_constraints_allows_failed_soft_constraint() -> None:
    results = an.validate_training_constraints(
        6,
        [ValueAtMostConstraint(5, kind=an.ConstraintKind.SOFT)],
    )

    assert results[0].passed is False
    assert results[0].blocks_candidate is False


def test_validate_training_constraints_is_exported_from_package_root() -> None:
    assert (
        an.validate_training_constraints(3, [ValueAtMostConstraint(5)])[0].passed
        is True
    )
