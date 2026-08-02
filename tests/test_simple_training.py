from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest

import agentnet as an


class LookupModule(an.Module):
    def __init__(self, name: str, outputs: dict[object, object]) -> None:
        super().__init__(name)
        self.outputs = outputs

    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return self.outputs.get(input, input)


class FailingModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del input, context
        raise RuntimeError("credential-value must not escape")


class ProposalAwareLLM:
    name = "proposal-aware"
    model = "local/proposal-aware"

    def __init__(self) -> None:
        self.requests: list[an.ChatRequest] = []

    async def complete(self, request: an.ChatRequest) -> an.ChatResponse:
        self.requests.append(request)
        system = request.messages[0]["content"]
        if system.startswith("Improve agent instructions"):
            content = json.dumps(
                {
                    "prompts": [
                        {
                            "instructions": "improved classification instructions",
                            "target": "classifier",
                        }
                    ]
                }
            )
        elif system.startswith("Design one bounded specialist"):
            content = json.dumps(
                {
                    "reducer": "Return the primary answer.",
                    "specialist": "Check the primary answer.",
                }
            )
        else:
            content = "yes" if "improved" in system else "no"
        return an.ChatResponse(content, model=request.model)

    async def stream(self, request: an.ChatRequest) -> AsyncIterator[an.ChatEvent]:
        yield an.ChatEvent.from_response(await self.complete(request))


def test_train_accepts_plain_x_y_and_returns_runnable_network() -> None:
    net = LookupModule("normalizer", {"a": " YES ", "b": "no"})

    fitted = an.train(net, ["a", "b", "a", "b", "a"], ["yes", "no", "yes", "no", "yes"])

    assert isinstance(fitted, an.FittedAgentNet)
    assert fitted is not net
    assert fitted.run("a") == " YES "
    assert fitted.training.best_score == 1.0
    assert len(fitted.training.validation_indices) == 1
    assert fitted.training.optimizer == "AutoOptimizer"


def test_train_validates_xy_and_inferred_labels() -> None:
    net = LookupModule("echo", {})

    with pytest.raises(an.AgentNetValidationError, match="different numbers"):
        an.train(net, ["a", "b"], ["a"], optimize="none")

    with pytest.raises(an.AgentNetConfigurationError, match="expected_output is missing"):
        an.train(
            net,
            an.Dataset([an.TrainingExample("a")]),
            optimize="none",
        )


def test_explicit_candidates_rank_on_validation_and_keep_input_unchanged() -> None:
    seed = LookupModule("seed", {"train": "yes", "validation": "no"})
    candidate = LookupModule("candidate", {"train": "no", "validation": "yes"})

    fitted = an.train(
        seed,
        ["train"],
        ["yes"],
        validation_data=(["validation"], ["yes"]),
        optimize=an.ExplicitCandidates([candidate]),
    )

    assert fitted.net is candidate
    assert fitted.training.train_score == 0.0
    assert fitted.training.validation_score == 1.0
    assert seed.outputs["validation"] == "no"


def test_topology_optimizer_evaluates_live_module_candidates() -> None:
    seed = LookupModule("seed", {"case": "wrong"})
    replacement = LookupModule("replacement", {"case": "right"})
    optimizer = an.TopologyOptimizer(
        search_space=an.TopologySearchSpace(
            allowed_mutations=["node_replacement"],
            replacement_candidates=[replacement],
            max_trials=2,
        )
    )

    fitted = an.train(seed, ["case"], ["right"], optimize=optimizer)

    assert fitted.net is replacement
    assert fitted.training.best_trial.changes == ("topology:node_replacement",)


def test_failed_candidate_is_sanitized_and_search_continues() -> None:
    seed = LookupModule("seed", {"case": "wrong"})
    good = LookupModule("good", {"case": "right"})

    fitted = an.train(
        seed,
        ["case"],
        ["right"],
        optimize=an.ExplicitCandidates([FailingModule("failed"), good]),
    )

    assert fitted.net is good
    failed = next(trial for trial in fitted.training.trials if trial.status == "failed")
    assert failed.error_type == "RuntimeError"
    serialized = json.dumps(fitted.training.to_dict())
    assert "credential-value" not in serialized


def test_failed_baseline_can_be_replaced_by_a_successful_candidate() -> None:
    good = LookupModule("good", {"case": "right"})

    fitted = an.train(
        FailingModule("failed-baseline"),
        ["case"],
        ["right"],
        optimize=an.ExplicitCandidates([good]),
    )

    assert fitted.net is good
    assert fitted.training.trials[0].status == "failed"


def test_all_failed_candidates_raise() -> None:
    with pytest.raises(an.AgentNetValidationError, match="No training candidate"):
        an.train(
            FailingModule("failed-baseline"),
            ["case"],
            ["right"],
            optimize=an.ExplicitCandidates([FailingModule("also-failed")]),
        )


def test_equal_scores_prefer_the_simpler_network() -> None:
    seed = LookupModule("seed", {})
    complex_candidate = an.Sequential(LookupModule("first", {}), LookupModule("second", {}))

    fitted = an.train(
        seed,
        ["payload"],
        ["payload"],
        optimize=an.ExplicitCandidates([complex_candidate]),
    )

    assert fitted.net is seed


def test_auto_optimizer_improves_prompt_without_mutating_seed() -> None:
    llm = ProposalAwareLLM()
    seed = an.ReActAgent(
        "classifier",
        instructions="bad classification instructions",
        llms=[llm],
    )

    fitted = an.train(
        seed,
        ["case-1", "case-2", "case-3", "case-4", "case-5"],
        ["yes"] * 5,
        optimizer_llm=llm,
    )

    assert isinstance(fitted.net, an.ReActAgent)
    assert fitted.net.instructions == "improved classification instructions"
    assert seed.instructions == "bad classification instructions"
    assert fitted.training.best_trial.stage == "prompt"
    assert fitted.training.best_score == 1.0

    topology_trial = next(
        trial for trial in fitted.training.trials if trial.stage == "topology"
    )
    assert isinstance(topology_trial.net, an.Parallel)
    specialist = topology_trial.net.modules[-1]
    assert isinstance(specialist, an.ReActAgent)
    assert specialist.tools == ()
    assert isinstance(topology_trial.net.reducer, an.ReActAgent)
    assert topology_trial.net.reducer.tools == ()


def test_budget_rejects_baseline_before_execution() -> None:
    llm = ProposalAwareLLM()
    net = an.ReActAgent("classifier", instructions="bad", llms=[llm])

    with pytest.raises(an.AgentNetValidationError, match="baseline exceeds"):
        an.train(
            net,
            ["a", "b", "c", "d", "e"],
            ["yes"] * 5,
            budget=an.Budget(max_trials=10, max_llm_calls=2),
        )

    assert llm.requests == []


def test_async_training_and_fitted_evaluation() -> None:
    net = LookupModule("echo", {})

    async def run() -> tuple[an.FittedAgentNet, an.EvaluationResult]:
        fitted = await an.atrain(net, ["a"], ["a"], optimize="none")
        evaluation = await fitted.aevaluate(["b"], ["b"])
        return fitted, evaluation

    fitted, evaluation = anyio.run(run)

    assert fitted.training.best_score == 1.0
    assert evaluation.score == 1.0


def test_validation_split_is_deterministic_and_stratified_when_possible() -> None:
    X = list(range(10))
    y = ["even" if value % 2 == 0 else "odd" for value in X]
    net = LookupModule("classifier", dict(zip(X, y, strict=True)))

    first = an.train(net, X, y, optimize="none", random_state=7)
    second = an.train(net, X, y, optimize="none", random_state=7)

    assert first.training.validation_indices == second.training.validation_indices
    held_out = [y[index] for index in first.training.validation_indices]
    assert sorted(held_out) == ["even", "odd"]


def test_progress_and_report_serialization_are_prompt_and_data_free() -> None:
    events: list[an.TrainingTrialEvent] = []
    private_input = "private-example-value"
    net = LookupModule("echo", {private_input: "right"})

    fitted = an.train(
        net,
        [private_input],
        ["right"],
        optimize="none",
        progress=events.append,
    )

    payload = json.dumps(
        {
            "events": [event.to_dict() for event in events],
            "report": fitted.training.to_dict(),
        }
    )
    assert private_input not in payload
    assert events[0].net_descriptor == {"name": "echo", "type": "LookupModule"}


def test_fitted_network_saves_selected_training_history(tmp_path: Path) -> None:
    path = tmp_path / "trained.agentnet"
    net = an.ReActAgent(
        "classifier",
        instructions="Return yes.",
        llms=[an.FakeLLM(["yes"], name="local")],
    )
    fitted = an.train(net, ["case"], ["yes"], optimize="none")

    artifact = fitted.save(path)

    assert artifact.path == path
    assert artifact.training_history is not None
    assert len(artifact.training_history["steps"]) == 1


def test_simple_training_public_exports_are_available() -> None:
    assert an.train is not None
    assert an.atrain is not None
    assert an.AutoOptimizer is not None
    assert an.ExplicitCandidates is not None
    assert an.FittedAgentNet is not None
    assert an.TrainingReport is not None
    assert an.TrainingTrial is not None
    assert an.ExpectedOutputObjective is not None
