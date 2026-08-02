import pytest

import agentnet as an


def test_validate_semantic_equivalence_accepts_equal_values() -> None:
    result = an.validate_semantic_equivalence(
        {"summary": "ok"},
        {"summary": "ok"},
        source_representation="json",
        target_representation="yaml",
    )

    assert result.equivalent is True
    assert result.source_representation == "json"
    assert result.target_representation == "yaml"


def test_validate_semantic_equivalence_normalizes_text_whitespace() -> None:
    result = an.validate_semantic_equivalence("  same meaning\n", "same meaning")

    assert result.equivalent is True


def test_validate_semantic_equivalence_accepts_custom_comparator() -> None:
    result = an.validate_semantic_equivalence(
        "# Summary\nok",
        {"summary": "ok"},
        source_representation="markdown",
        target_representation="key_value",
        comparator=lambda source, target: "ok" in source
        and target["summary"] == "ok",
    )

    assert result.equivalent is True


def test_validate_semantic_equivalence_rejects_different_values() -> None:
    with pytest.raises(an.AgentNetValidationError, match="semantically equivalent"):
        an.validate_semantic_equivalence(
            {"summary": "ok"},
            {"summary": "changed"},
            source_representation="json",
            target_representation="json",
        )


def test_validate_semantic_equivalence_rejects_non_callable_comparator() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="comparator"):
        an.validate_semantic_equivalence("a", "a", comparator=object())


def test_semantic_equivalence_validation_is_exported_from_package_root() -> None:
    assert an.SemanticEquivalenceValidation is not None
    assert an.validate_semantic_equivalence is not None
