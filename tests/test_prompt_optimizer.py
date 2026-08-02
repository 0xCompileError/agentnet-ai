import pytest

import agentnet as an
from agentnet.constraints import Constraint


class PromptContainsConstraint(Constraint):
    def __init__(
        self,
        text: str,
        *,
        kind: an.ConstraintKind = an.ConstraintKind.HARD,
    ) -> None:
        super().__init__(f"prompt_contains_{text}", kind=kind)
        self.text = text

    def check(self, candidate: object, context: object | None = None) -> bool:
        return isinstance(candidate, str) and self.text in candidate


def test_prompt_optimizer_selects_highest_scoring_prompt() -> None:
    optimizer = an.PromptOptimizer()

    result = optimizer.optimize(
        ["Be brief.", "Be precise and cite sources."],
        scorer=lambda prompt: float(len(prompt)),
    )

    assert result.prompt == "Be precise and cite sources."
    assert result.candidate == result.prompt
    assert result.score == float(len(result.prompt))


def test_prompt_optimizer_skips_empty_prompts_by_default() -> None:
    optimizer = an.PromptOptimizer()

    result = optimizer.optimize(
        ["   ", "Use the evidence."],
        scorer=lambda prompt: 1.0,
    )

    assert result.prompt == "Use the evidence."
    assert result.metadata["rejected_candidates"] == 1


def test_prompt_optimizer_applies_hard_training_constraints() -> None:
    optimizer = an.PromptOptimizer(
        constraints=[PromptContainsConstraint("cite")],
    )

    result = optimizer.optimize(
        ["Answer directly.", "Answer and cite sources."],
        scorer=lambda prompt: 10.0 if prompt.startswith("Answer directly") else 1.0,
    )

    assert result.prompt == "Answer and cite sources."
    assert result.constraint_results[0].passed is True


def test_prompt_optimizer_allows_soft_constraint_violations_by_score() -> None:
    optimizer = an.PromptOptimizer(
        constraints=[
            PromptContainsConstraint("cite", kind=an.ConstraintKind.SOFT),
        ],
    )

    result = optimizer.optimize(
        ["Answer directly.", "Answer and cite sources."],
        scorer=lambda prompt: 10.0 if prompt == "Answer directly." else 1.0,
    )

    assert result.prompt == "Answer directly."
    assert result.constraint_results[0].passed is False


def test_prompt_optimizer_rejects_when_no_prompt_is_valid() -> None:
    optimizer = an.PromptOptimizer()

    with pytest.raises(an.AgentNetValidationError, match="No prompt candidate"):
        optimizer.optimize(["", "   "], scorer=lambda prompt: 1.0)


def test_prompt_optimizer_reports_final_candidate_counts() -> None:
    optimizer = an.PromptOptimizer(
        constraints=[PromptContainsConstraint("cite")],
        metadata={"optimizer": "prompt"},
    )

    result = optimizer.optimize(
        ["cite facts", "omit sources", "cite sources thoroughly"],
        scorer=lambda prompt: float(len(prompt)),
    )

    assert result.prompt == "cite sources thoroughly"
    assert result.metadata["optimizer"] == "prompt"
    assert result.metadata["evaluated_candidates"] == 2
    assert result.metadata["rejected_candidates"] == 1
    assert result.metadata["training_constraint_results"] == [
        {
            "blocks_candidate": False,
            "constraint": "prompt_contains_cite",
            "kind": "hard",
            "message": None,
            "passed": True,
        }
    ]


def test_prompt_optimizer_is_exported_from_package_root() -> None:
    assert an.PromptOptimizer is not None
    assert an.PromptOptimizationResult is not None
