import agentnet as an


def test_detect_lossy_translation_reports_preserved_fields_as_not_lossy() -> None:
    result = an.detect_lossy_translation(
        {"summary": "ok", "details": "extra"},
        {"summary": "ok"},
        source_representation="json",
        target_representation="markdown",
        required_fields=["summary"],
    )

    assert result.lossy is False
    assert result.losses == ()
    assert result.source_representation == "json"
    assert result.target_representation == "markdown"


def test_detect_lossy_translation_reports_missing_required_field() -> None:
    result = an.detect_lossy_translation(
        {"summary": "ok"},
        {"details": "missing"},
        required_fields=["summary"],
    )

    assert result.lossy is True
    assert "summary" in result.losses[0]


def test_detect_lossy_translation_reports_changed_required_field() -> None:
    result = an.detect_lossy_translation(
        {"summary": "ok"},
        {"summary": "changed"},
        required_fields=["summary"],
    )

    assert result.lossy is True
    assert "summary" in result.losses[0]


def test_detect_lossy_translation_uses_whole_value_equivalence_by_default() -> None:
    result = an.detect_lossy_translation("same", "different")

    assert result.lossy is True
    assert "semantically equivalent" in result.losses[0]


def test_detect_lossy_translation_accepts_custom_comparator() -> None:
    result = an.detect_lossy_translation(
        {"summary": "OK"},
        {"summary": "ok"},
        required_fields=["summary"],
        comparator=lambda source, target: source.lower() == target.lower(),
    )

    assert result.lossy is False


def test_lossy_translation_detection_is_exported_from_package_root() -> None:
    assert an.LossyTranslationDetection is not None
    assert an.detect_lossy_translation is not None
