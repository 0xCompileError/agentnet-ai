import pytest

import agentnet as an


def test_information_transfer_optimizer_selects_highest_scoring_valid_payload() -> None:
    target = an.Interface(schema=an.Schema({"summary": str}))
    optimizer = an.InformationTransferOptimizer()

    result = optimizer.optimize(
        [{"summary": "short"}, {"summary": "longer summary"}],
        target=target,
        scorer=lambda value: float(len(value["summary"])),
    )

    assert result.value == {"summary": "longer summary"}
    assert result.candidate == result.value
    assert result.score == float(len("longer summary"))


def test_information_transfer_optimizer_rejects_invalid_target_payloads() -> None:
    target = an.Interface(
        semantic_contract=an.SemanticContract(required_fields=["summary"])
    )
    optimizer = an.InformationTransferOptimizer()

    result = optimizer.optimize(
        [{"details": "missing"}, {"summary": "ok"}],
        target=target,
        scorer=lambda value: 1.0,
    )

    assert result.value == {"summary": "ok"}
    assert result.metadata["rejected_candidates"] == 1


def test_information_transfer_optimizer_validates_selected_representation() -> None:
    target = an.Interface(
        representations=[
            an.KeyValueRepresentation(required_keys=["summary"]),
        ]
    )
    optimizer = an.InformationTransferOptimizer()

    result = optimizer.optimize(
        [{"details": "missing"}, {"summary": "ok"}],
        target=target,
        representation="key_value",
        scorer=lambda value: 1.0,
    )

    assert result.value == {"summary": "ok"}
    assert result.representation == "key_value"


def test_information_transfer_optimizer_applies_constraints() -> None:
    target = an.Interface(
        representations=[an.KeyValueRepresentation(required_keys=["summary"])]
    )
    optimizer = an.InformationTransferOptimizer(
        constraints=[an.RepresentationConstraint(["key_value"])]
    )

    result = optimizer.optimize(
        [{"summary": "ok"}],
        target=target,
        representation="key_value",
        scorer=lambda value: 1.0,
    )

    assert result.constraint_results[0].passed is True


def test_information_transfer_optimizer_rejects_when_no_payload_is_valid() -> None:
    target = an.Interface(schema=an.Schema({"summary": str}))
    optimizer = an.InformationTransferOptimizer()

    with pytest.raises(an.AgentNetValidationError, match="No information transfer"):
        optimizer.optimize(
            [{"summary": 3}],
            target=target,
            scorer=lambda value: 1.0,
        )


def test_information_transfer_optimizer_reports_final_candidate_counts() -> None:
    target = an.Interface(
        representations=[an.KeyValueRepresentation(required_keys=["summary"])]
    )
    optimizer = an.InformationTransferOptimizer(
        constraints=[an.RepresentationConstraint(["key_value"])],
        metadata={"optimizer": "information-transfer"},
    )

    result = optimizer.optimize(
        [{"details": "missing"}, {"summary": "ok"}],
        target=target,
        representation="key_value",
        scorer=lambda value: 1.0,
    )

    assert result.metadata["optimizer"] == "information-transfer"
    assert result.metadata["evaluated_candidates"] == 1
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


def test_information_transfer_optimizer_is_exported_from_package_root() -> None:
    assert an.InformationTransferOptimizer is not None
    assert an.InformationTransferOptimizationResult is not None
