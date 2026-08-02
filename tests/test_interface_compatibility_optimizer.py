import pytest

import agentnet as an


def test_interface_compatibility_optimizer_selects_best_compatible_target() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary", "risks"]),
        representations=[an.Representation("json"), an.Representation("markdown")],
    )
    json_target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )
    markdown_target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("markdown")],
    )
    optimizer = an.InterfaceCompatibilityOptimizer()

    result = optimizer.optimize(
        source,
        [json_target, markdown_target],
        scorer=lambda compatibility: 10.0
        if compatibility.negotiation.identifier == "markdown"
        else 1.0,
    )

    assert result.compatibility.target is markdown_target
    assert result.representation == "markdown"
    assert result.candidate is result.compatibility
    assert result.score == 10.0


def test_interface_compatibility_optimizer_rejects_incompatible_targets() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )
    incompatible = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary", "risk"]),
        representations=[an.Representation("json")],
    )
    optimizer = an.InterfaceCompatibilityOptimizer()

    with pytest.raises(an.AgentNetValidationError, match="No interface compatibility"):
        optimizer.optimize(source, [incompatible], scorer=lambda compatibility: 1.0)


def test_interface_compatibility_optimizer_applies_constraints() -> None:
    source = an.Interface(
        representations=[an.Representation("json"), an.Representation("markdown")]
    )
    json_target = an.Interface(representations=[an.Representation("json")])
    markdown_target = an.Interface(representations=[an.Representation("markdown")])
    optimizer = an.InterfaceCompatibilityOptimizer(
        constraints=[an.RepresentationConstraint(["json"])]
    )

    result = optimizer.optimize(
        source,
        [json_target, markdown_target],
        scorer=lambda compatibility: 10.0
        if compatibility.negotiation.identifier == "markdown"
        else 1.0,
    )

    assert result.compatibility.target is json_target
    assert result.constraint_results[0].passed is True
    assert result.metadata["rejected_candidates"] == 1


def test_interface_compatibility_optimizer_reports_final_candidate_counts() -> None:
    source = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary", "risks"]),
        representations=[an.Representation("json"), an.Representation("markdown")],
    )
    json_target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"]),
        representations=[an.Representation("json")],
    )
    incompatible_target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["extra"]),
        representations=[an.Representation("json")],
    )
    markdown_target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["risks"]),
        representations=[an.Representation("markdown")],
    )
    optimizer = an.InterfaceCompatibilityOptimizer(
        constraints=[an.RepresentationConstraint(["markdown"])],
        metadata={"optimizer": "interface-compatibility"},
    )

    result = optimizer.optimize(
        source,
        [json_target, incompatible_target, markdown_target],
        scorer=lambda compatibility: 10.0
        if compatibility.negotiation.identifier == "markdown"
        else 1.0,
    )

    assert result.compatibility.target is markdown_target
    assert result.metadata["optimizer"] == "interface-compatibility"
    assert result.metadata["evaluated_candidates"] == 1
    assert result.metadata["rejected_candidates"] == 2
    assert result.metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "representation",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_interface_compatibility_optimizer_is_exported_from_package_root() -> None:
    assert an.InterfaceCompatibilityOptimizer is not None
    assert an.InterfaceCompatibilityOptimizationResult is not None
