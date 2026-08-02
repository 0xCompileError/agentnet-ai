import pytest

import agentnet as an


def test_translation_strategy_optimizer_selects_registered_translator_strategy() -> None:
    source = an.Interface(
        representations=[
            an.Representation("markdown"),
            an.Representation("json"),
        ]
    )
    target = an.Interface(representations=[an.Representation("json")])
    registry = an.RepresentationTranslatorRegistry(
        [
            an.RepresentationTranslator(
                "markdown",
                "json",
                lambda value: {"text": value},
            )
        ]
    )
    optimizer = an.TranslationStrategyOptimizer()

    result = optimizer.optimize(
        source,
        target,
        translator_registry=registry,
        scorer=lambda strategy: 10.0 if strategy.mode == "translate" else 1.0,
    )

    assert result.strategy.mode == "translate"
    assert result.strategy.source_representation == "markdown"
    assert result.strategy.target_representation == "json"
    assert result.strategy.translate("hello") == {"text": "hello"}


def test_translation_strategy_optimizer_uses_identity_for_direct_match() -> None:
    source = an.Interface(representations=[an.Representation("json")])
    target = an.Interface(representations=[an.Representation("json")])
    optimizer = an.TranslationStrategyOptimizer()

    result = optimizer.optimize(
        source,
        target,
        scorer=lambda strategy: 1.0,
    )

    assert result.strategy.mode == "identity"
    assert result.strategy.representation == "json"
    assert result.candidate is result.strategy
    assert result.strategy.translate({"ok": True}) == {"ok": True}


def test_translation_strategy_optimizer_applies_representation_constraints() -> None:
    source = an.Interface(
        representations=[
            an.Representation("markdown"),
            an.Representation("xml"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("yaml"),
        ]
    )
    registry = an.RepresentationTranslatorRegistry(
        [
            an.RepresentationTranslator("markdown", "json", lambda value: value),
            an.RepresentationTranslator("xml", "yaml", lambda value: value),
        ]
    )
    optimizer = an.TranslationStrategyOptimizer(
        constraints=[an.RepresentationConstraint(["yaml"])]
    )

    result = optimizer.optimize(
        source,
        target,
        translator_registry=registry,
        scorer=lambda strategy: 10.0
        if strategy.target_representation == "json"
        else 1.0,
    )

    assert result.strategy.target_representation == "yaml"
    assert result.constraint_results[0].passed is True
    assert result.metadata["rejected_candidates"] == 1


def test_translation_strategy_optimizer_rejects_when_no_strategy_exists() -> None:
    source = an.Interface(representations=[an.Representation("markdown")])
    target = an.Interface(representations=[an.Representation("json")])
    optimizer = an.TranslationStrategyOptimizer()

    with pytest.raises(an.AgentNetValidationError, match="No translation strategy"):
        optimizer.optimize(source, target, scorer=lambda strategy: 1.0)


def test_translation_strategy_optimizer_reports_final_candidate_counts() -> None:
    source = an.Interface(
        representations=[
            an.Representation("markdown"),
            an.Representation("xml"),
        ]
    )
    target = an.Interface(
        representations=[
            an.Representation("json"),
            an.Representation("yaml"),
        ]
    )
    registry = an.RepresentationTranslatorRegistry(
        [
            an.RepresentationTranslator("markdown", "json", lambda value: value),
            an.RepresentationTranslator("xml", "yaml", lambda value: value),
        ]
    )
    optimizer = an.TranslationStrategyOptimizer(
        constraints=[an.RepresentationConstraint(["json"])],
        metadata={"optimizer": "translation-strategy"},
    )

    result = optimizer.optimize(
        source,
        target,
        translator_registry=registry,
        scorer=lambda strategy: 10.0
        if strategy.target_representation == "json"
        else 1.0,
    )

    assert result.strategy.target_representation == "json"
    assert result.metadata["optimizer"] == "translation-strategy"
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


def test_translation_strategy_optimizer_is_exported_from_package_root() -> None:
    assert an.TranslationStrategy is not None
    assert an.TranslationStrategyOptimizer is not None
    assert an.TranslationStrategyOptimizationResult is not None
