import pytest

import agentnet as an


def test_representation_selection_optimizer_selects_highest_scored_representation() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    optimizer = an.RepresentationSelectionOptimizer()

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda representation: 10.0
        if representation.identifier == "markdown"
        else 1.0,
    )

    assert result.representation.identifier == "markdown"
    assert result.identifier == "markdown"
    assert result.candidate is result.representation
    assert result.score == 10.0


def test_representation_selection_optimizer_applies_representation_constraints() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    optimizer = an.RepresentationSelectionOptimizer(
        constraints=[an.RepresentationConstraint(["json"])]
    )

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda representation: 10.0
        if representation.identifier == "markdown"
        else 1.0,
    )

    assert result.representation.identifier == "json"
    assert result.constraint_results[0].passed is True
    assert result.metadata["rejected_candidates"] == 1


def test_representation_selection_optimizer_allows_soft_constraint_violations() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
        ]
    )
    optimizer = an.RepresentationSelectionOptimizer(
        constraints=[
            an.RepresentationConstraint(
                ["json"],
                kind=an.ConstraintKind.SOFT,
            )
        ]
    )

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda representation: 10.0
        if representation.identifier == "markdown"
        else 1.0,
    )

    assert result.representation.identifier == "markdown"
    assert result.constraint_results[0].passed is False


def test_representation_selection_optimizer_rejects_incompatible_interfaces() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("xml")])
    optimizer = an.RepresentationSelectionOptimizer()

    with pytest.raises(an.AgentNetValidationError, match="No representation selection"):
        optimizer.optimize(source, target, scorer=lambda representation: 1.0)


def test_representation_selection_optimizer_reports_final_candidate_counts() -> None:
    source = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
            an.Representation("xml"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("markdown"),
            an.Representation("xml"),
        ]
    )
    optimizer = an.RepresentationSelectionOptimizer(
        constraints=[an.RepresentationConstraint(["json", "markdown"])],
        metadata={"optimizer": "representation-selection"},
    )

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda representation: 10.0
        if representation.identifier == "markdown"
        else 1.0,
    )

    assert result.representation.identifier == "markdown"
    assert result.metadata["optimizer"] == "representation-selection"
    assert result.metadata["evaluated_candidates"] == 2
    assert result.metadata["rejected_candidates"] == 1
    assert result.metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "representation",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_representation_selection_optimizer_is_exported_from_package_root() -> None:
    assert an.RepresentationSelectionOptimizer is not None
    assert an.RepresentationSelectionOptimizationResult is not None
