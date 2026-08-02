import pytest

import agentnet as an
from agentnet.constraints import Constraint


class AtMostConstraint(Constraint):
    def __init__(
        self,
        max_value: int,
        *,
        kind: an.ConstraintKind = an.ConstraintKind.HARD,
    ) -> None:
        super().__init__(f"at_most_{max_value}", kind=kind)
        self.max_value = max_value

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, int) and candidate <= self.max_value


def test_constraint_aware_optimizer_skips_hard_invalid_candidates() -> None:
    scored: list[int] = []
    optimizer = an.ConstraintAwareOptimizer(constraints=[AtMostConstraint(5)])

    result = optimizer.optimize(
        [10, 3],
        scorer=lambda candidate: scored.append(candidate) or float(candidate),
    )

    assert result.candidate == 3
    assert result.score == 3.0
    assert scored == [3]
    assert result.metadata["rejected_candidates"] == 1


def test_constraint_aware_optimizer_allows_soft_invalid_candidates_by_score() -> None:
    optimizer = an.ConstraintAwareOptimizer(
        constraints=[AtMostConstraint(5, kind=an.ConstraintKind.SOFT)]
    )

    result = optimizer.optimize(
        [10, 3],
        scorer=float,
    )

    assert result.candidate == 10
    assert result.constraint_results[0].passed is False
    assert result.constraint_results[0].blocks_candidate is False


def test_constraint_aware_optimizer_rejects_when_no_candidate_satisfies_hard_constraints() -> None:
    optimizer = an.ConstraintAwareOptimizer(constraints=[AtMostConstraint(5)])

    with pytest.raises(an.AgentNetValidationError, match="No candidate"):
        optimizer.optimize([10, 20], scorer=float)


def test_constraint_aware_optimizer_reports_final_candidate_counts() -> None:
    optimizer = an.ConstraintAwareOptimizer(
        constraints=[AtMostConstraint(5)],
        metadata={"optimizer": "constraint-aware"},
    )

    result = optimizer.optimize(
        [4, 10, 5],
        scorer=lambda candidate: 10.0 if candidate == 4 else 1.0,
    )

    assert result.candidate == 4
    assert result.metadata["optimizer"] == "constraint-aware"
    assert result.metadata["evaluated_candidates"] == 2
    assert result.metadata["rejected_candidates"] == 1
    assert result.metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "at_most_5",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_constraint_aware_optimizer_is_exported_from_package_root() -> None:
    assert an.ConstraintAwareOptimizer(constraints=[]).optimize([1], scorer=float).candidate == 1
