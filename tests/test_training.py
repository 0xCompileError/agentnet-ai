import json

import pytest

import agentnet as an


class EchoModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        return input


class PromptHolder:
    def __init__(self) -> None:
        self.instructions = "Be brief."


class CountingModule(an.Module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls = 0

    async def arun(self, input: object, context: object | None = None) -> object:
        del context
        self.calls += 1
        return input


class PrivateCandidate(EchoModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.instructions = "private candidate prompt"
        self.api_key = "credential-value"
        self.client = object()


def test_dataset_abstraction_defensively_serializes_examples() -> None:
    metadata = {"split": "train"}
    example = an.TrainingExample(
        input={"question": "Ship?"},
        expected_output={"answer": "yes"},
        id="case-1",
        metadata=metadata,
    )
    metadata["split"] = "changed"
    dataset = an.Dataset([example], name="decisions", metadata={"owner": "eval"})

    assert len(dataset) == 1
    assert dataset[0].metadata == {"split": "train"}
    assert dataset.to_dict() == {
        "examples": [
            {
                "expected_output": {"answer": "yes"},
                "id": "case-1",
                "input": {"question": "Ship?"},
                "metadata": {"split": "train"},
            }
        ],
        "metadata": {"owner": "eval"},
        "name": "decisions",
    }
    assert an.Dataset.from_dict(dataset.to_dict()).to_dict() == dataset.to_dict()


def test_dataset_rejects_secret_like_metadata_keys() -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="may serialize secrets"):
        an.Dataset([], metadata={"api_token": "secret"})


def test_trainer_scores_dataset_and_records_history_and_checkpoints() -> None:
    dataset = an.Dataset(
        [
            an.TrainingExample("alpha", expected_output="alpha", id="a"),
            an.TrainingExample("beta", expected_output="beta", id="b"),
        ],
        name="echoes",
    )
    trainer = an.Trainer(
        net=EchoModule("echo"),
        objective=an.ExactMatchObjective("alpha"),
        budget=an.Budget(max_epochs=1, max_examples=2, max_trials=1),
    )

    result = trainer.fit(dataset, epochs=1)

    assert result.net.name == "echo"
    assert result.score == 0.5
    assert result.objective_result.metrics["training.example_count"] == 2.0
    assert result.objective_result.passed is False
    assert [step.example_id for step in result.history.steps] == ["a", "b"]
    assert result.history.best_step.score == 1.0
    assert result.checkpoints[0].epoch == 1
    assert result.checkpoints[0].score == 0.5
    assert len(result.candidate_results) == 1
    assert result.candidate_results[0].candidate is result.net
    assert result.candidate_results[0].checkpoint is result.checkpoints[0]


def test_trainer_chooses_best_candidate_network() -> None:
    dataset = an.Dataset(
        [
            an.TrainingExample("payload", expected_output="payload", id="one"),
            an.TrainingExample("other", expected_output="payload", id="two"),
        ]
    )
    objective = an.ExactMatchObjective("payload")
    first = EchoModule("first")
    second = EchoModule("second")

    result = an.Trainer(net=EchoModule("base"), objective=objective).fit(
        dataset,
        candidates=[first, second],
        candidate_scorer=lambda candidate, evaluation, history: (
            evaluation.score + (1.0 if candidate.name == "second" else 0.0)
        ),
    )

    assert result.net is second
    assert result.score == 1.5
    assert result.metadata["evaluated_candidates"] == 2
    assert [item.candidate for item in result.candidate_results] == [first, second]
    assert [item.score for item in result.candidate_results] == [0.5, 1.5]
    assert result.candidate_results[0].history is not result.candidate_results[1].history
    assert [
        [step.example_id for step in item.history.steps]
        for item in result.candidate_results
    ] == [["one", "two"], ["one", "two"]]
    assert [checkpoint.score for checkpoint in result.checkpoints] == [0.5, 1.5]
    assert result.history is result.candidate_results[1].history
    assert result.objective_result is result.candidate_results[1].objective_result


def test_trainer_reports_ties_and_keeps_first_candidate_selection() -> None:
    first = EchoModule("first")
    second = EchoModule("second")

    result = an.Trainer(
        net=first,
        objective=an.ExactMatchObjective("payload"),
    ).fit(
        [an.TrainingExample("payload")],
        candidates=[first, second],
    )

    assert result.net is first
    assert result.is_tied is True
    assert result.tied_candidates == result.candidate_results
    assert result.metadata["selection_policy"] == "first"
    assert result.to_dict()["tied_candidates"] == [
        {"name": "first", "type": "EchoModule"},
        {"name": "second", "type": "EchoModule"},
    ]


def test_training_result_serializes_all_candidates_without_live_state() -> None:
    first = PrivateCandidate("private-first")
    second = PrivateCandidate("private-second")

    result = an.Trainer(
        net=first,
        objective=an.ExactMatchObjective("input payload"),
    ).fit(
        [an.TrainingExample("input payload", id="case-1")],
        candidates=[first, second],
        candidate_scorer=lambda candidate, evaluation, history: (
            evaluation.score + (0.5 if candidate is second else 0.0)
        ),
    )

    serialized = result.to_dict()
    serialized_text = json.dumps(serialized)

    assert [
        candidate["candidate_descriptor"]
        for candidate in serialized["candidate_results"]
    ] == [
        {"name": "private-first", "type": "PrivateCandidate"},
        {"name": "private-second", "type": "PrivateCandidate"},
    ]
    assert len(serialized["checkpoints"]) == 2
    assert "private candidate prompt" not in serialized_text
    assert "credential-value" not in serialized_text
    assert "client" not in serialized_text
    assert "input payload" not in serialized_text


def test_trainer_emits_ordered_descriptor_safe_progress_events() -> None:
    events: list[an.TrainingProgressEvent] = []
    first = EchoModule("first")
    second = EchoModule("second")

    result = an.Trainer(
        net=first,
        objective=an.ExactMatchObjective("alpha"),
    ).fit(
        [
            an.TrainingExample("alpha", id="a"),
            an.TrainingExample("beta", id="b"),
        ],
        candidates=[first, second],
        progress_callback=events.append,
    )

    assert [event.event_type for event in events] == [
        "training.started",
        "candidate.started",
        "example.started",
        "example.completed",
        "example.started",
        "example.completed",
        "candidate.completed",
        "candidate.started",
        "example.started",
        "example.completed",
        "example.started",
        "example.completed",
        "candidate.completed",
        "training.completed",
    ]
    assert all(event.candidate_count == 2 for event in events)
    assert all(event.example_count == 2 for event in events)
    assert [
        (event.candidate_index, event.example_index, event.score, event.passed)
        for event in events
        if event.event_type == "example.completed"
    ] == [
        (0, 0, 1.0, True),
        (0, 1, 0.0, False),
        (1, 0, 1.0, True),
        (1, 1, 0.0, False),
    ]
    assert [
        event.score
        for event in events
        if event.event_type == "candidate.completed"
    ] == [0.5, 0.5]
    assert events[-1].score == result.score
    assert events[-1].candidate_descriptor == {
        "name": "first",
        "type": "EchoModule",
    }
    serialized_events = json.dumps([event.to_dict() for event in events])
    assert "alpha" not in serialized_events
    assert "beta" not in serialized_events


def test_trainer_propagates_progress_callback_errors_before_running_example() -> None:
    candidate = CountingModule("counting")
    event_types: list[str] = []

    def fail_on_example(event: an.TrainingProgressEvent) -> None:
        event_types.append(event.event_type)
        if event.event_type == "example.started":
            raise RuntimeError("monitor unavailable")

    with pytest.raises(RuntimeError, match="monitor unavailable"):
        an.Trainer(
            net=candidate,
            objective=an.ExactMatchObjective("payload"),
        ).fit(
            [an.TrainingExample("payload")],
            progress_callback=fail_on_example,
        )

    assert event_types == [
        "training.started",
        "candidate.started",
        "example.started",
    ]
    assert candidate.calls == 0


def test_training_progress_event_round_trips_descriptor_only_payload() -> None:
    event = an.TrainingProgressEvent(
        "example.completed",
        candidate_count=2,
        example_count=10,
        epochs=1,
        candidate_index=1,
        example_index=4,
        epoch=1,
        example_id="case-05",
        candidate_descriptor={"name": "candidate", "type": "EchoModule"},
        score=0.75,
        passed=True,
    )

    assert event.candidate_number == 2
    assert event.example_number == 5
    assert an.TrainingProgressEvent.from_dict(event.to_dict()).to_dict() == event.to_dict()

    with pytest.raises(an.AgentNetConfigurationError, match="serialize secrets"):
        an.TrainingProgressEvent(
            "training.started",
            candidate_count=1,
            example_count=1,
            epochs=1,
            candidate_descriptor={"api_key": "credential-value"},
        )


def test_training_checkpoint_round_trips_without_serializing_live_network() -> None:
    checkpoint = an.TrainingCheckpoint(
        epoch=2,
        step=5,
        score=0.75,
        objective_result=an.EvaluationResult(score=0.75, passed=True),
        history=an.TrainingHistory(
            [
                an.TrainingStep(
                    epoch=2,
                    example_id="case-1",
                    score=0.75,
                    passed=True,
                )
            ]
        ),
        candidate_descriptor={"name": "decision-net"},
        metadata={"checkpoint": "manual"},
    )

    serialized = checkpoint.to_dict()

    assert serialized["candidate_descriptor"] == {"name": "decision-net"}
    assert "net" not in serialized
    assert an.TrainingCheckpoint.from_dict(serialized).to_dict() == serialized


def test_prompt_optimizer_remains_available_for_training_prompt_candidates() -> None:
    optimizer = an.PromptOptimizer()

    result = optimizer.optimize(
        ["Be brief.", "Be precise and cite sources."],
        scorer=lambda prompt: 1.0 if "cite" in prompt else 0.25,
    )

    assert result.prompt == "Be precise and cite sources."
    assert result.score == 1.0


def test_fallback_optimizer_selects_best_fallback_order() -> None:
    policy = an.LLMPolicy(["primary", "cheap", "strong"])
    optimizer = an.FallbackOptimizer()

    result = optimizer.optimize(
        policy,
        scorer=lambda candidate: (
            2.0 if candidate.candidates == ("primary", "strong", "cheap") else 1.0
        ),
    )

    assert result.policy.candidates == ("primary", "strong", "cheap")
    assert result.score == 2.0
    assert result.candidate is result.policy
    assert result.metadata["evaluated_candidates"] == 2


def test_retry_policy_optimizer_selects_best_retry_policy() -> None:
    conservative = an.RetryPolicy(transport_retries=0, quality_retries=0)
    resilient = an.RetryPolicy(transport_retries=2, quality_retries=1)
    optimizer = an.RetryPolicyOptimizer()

    result = optimizer.optimize(
        [conservative, resilient],
        scorer=lambda policy: float(policy.transport_retries + policy.quality_retries),
    )

    assert result.policy is resilient
    assert result.score == 3.0
    assert result.candidate is result.policy
    assert result.metadata["evaluated_candidates"] == 2


def test_attribution_engine_reports_score_and_metric_deltas_for_patch() -> None:
    patch = an.TrainingPatch(
        target="planner",
        field="instructions",
        kind="prompt",
        old_value="Be brief.",
        new_value="Cite evidence.",
        rationale="Improve evidence use.",
    )
    before = an.EvaluationResult(
        score=0.2,
        passed=False,
        metrics={"exact_match.score": 0.0},
    )
    after = an.EvaluationResult(
        score=0.9,
        passed=True,
        metrics={"exact_match.score": 1.0},
    )

    record = an.AttributionEngine().attribute(
        patch=patch,
        before=before,
        after=after,
        evidence=["case-1"],
    )

    assert record.patch_id == patch.id
    assert record.score_delta == pytest.approx(0.7)
    assert record.metric_deltas == {"exact_match.score": 1.0}
    assert record.evidence == ("case-1",)


def test_training_patch_generation_apply_and_rollback_for_object_attributes() -> None:
    target = PromptHolder()

    patch = an.generate_training_patch(
        target=target,
        target_name="planner",
        field="instructions",
        kind="prompt",
        new_value="Cite evidence.",
        rationale="Improve evidence use.",
    )

    assert patch.diff == {
        "field": "instructions",
        "new_value": "Cite evidence.",
        "old_value": "Be brief.",
    }
    patch.apply({"planner": target})
    assert target.instructions == "Cite evidence."
    patch.rollback({"planner": target})
    assert target.instructions == "Be brief."


def test_training_patch_generation_apply_and_rollback_for_mapping_targets() -> None:
    target: dict[str, object] = {"fallback_order": ["cheap", "strong"]}
    patch = an.generate_training_patch(
        target=target,
        target_name="policy",
        field="fallback_order",
        kind="fallback_order",
        new_value=["strong", "cheap"],
        rationale="Prefer stronger backup first.",
    )

    patch.apply({"policy": target})
    assert target["fallback_order"] == ["strong", "cheap"]
    patch.rollback({"policy": target})
    assert target["fallback_order"] == ["cheap", "strong"]


def test_training_history_tracks_best_step_and_round_trips() -> None:
    history = an.TrainingHistory()

    history.add(an.TrainingStep(epoch=1, example_id="a", score=0.25, passed=False))
    history.add(an.TrainingStep(epoch=1, example_id="b", score=0.75, passed=True))

    assert history.best_step.example_id == "b"
    assert history.best_score == 0.75
    assert an.TrainingHistory.from_dict(history.to_dict()).to_dict() == history.to_dict()


def test_budget_manager_tracks_limits_and_remaining_values() -> None:
    budget = an.Budget(max_epochs=2, max_examples=3, max_trials=4, max_cost=10.0)

    assert budget.can_run(epoch=1, examples=2, trials=1, cost=2.5)
    budget.record(examples=2, trials=1, cost=2.5)

    assert budget.examples_used == 2
    assert budget.remaining == {
        "cost": 7.5,
        "epochs": 2,
        "examples": 1,
        "llm_calls": None,
        "trials": 3,
    }
    assert not budget.can_run(epoch=3)
    assert not budget.can_run(examples=2)


def test_training_public_exports_are_available() -> None:
    assert an.Trainer is not None
    assert an.TrainingCandidateResult is not None
    assert an.TrainingResult is not None
    assert an.Dataset is not None
    assert an.TrainingExample is not None
    assert an.TrainingCheckpoint is not None
    assert an.TrainingPatch is not None
    assert an.TrainingProgressCallback is not None
    assert an.TrainingProgressEvent is not None
    assert an.AttributionEngine is not None
    assert an.FallbackOptimizer is not None
    assert an.RetryPolicyOptimizer is not None
    assert an.Budget is not None
