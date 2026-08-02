import pytest

import agentnet as an


def test_validate_information_preservation_accepts_required_fields() -> None:
    result = an.validate_information_preservation(
        {"summary": "ok", "details": "extra"},
        {"summary": "ok", "format": "markdown"},
        required_fields=["summary"],
    )

    assert result.preserved is True
    assert result.required_fields == ("summary",)


def test_validate_information_preservation_rejects_missing_required_field() -> None:
    with pytest.raises(an.AgentNetValidationError, match="summary"):
        an.validate_information_preservation(
            {"summary": "ok"},
            {"details": "missing"},
            required_fields=["summary"],
        )


def test_validate_information_preservation_rejects_changed_required_field() -> None:
    with pytest.raises(an.AgentNetValidationError, match="summary"):
        an.validate_information_preservation(
            {"summary": "ok"},
            {"summary": "changed"},
            required_fields=["summary"],
        )


def test_validate_information_preservation_accepts_custom_comparator() -> None:
    result = an.validate_information_preservation(
        {"summary": "OK"},
        {"summary": "ok"},
        required_fields=["summary"],
        comparator=lambda source, target: source.lower() == target.lower(),
    )

    assert result.preserved is True


def test_validate_information_preservation_defaults_to_whole_value_equivalence() -> None:
    result = an.validate_information_preservation(
        "  same information ",
        "same information",
    )

    assert result.required_fields == ()
    assert result.preserved is True


def test_information_preservation_validation_is_exported_from_package_root() -> None:
    assert an.InformationPreservationValidation is not None
    assert an.validate_information_preservation is not None
