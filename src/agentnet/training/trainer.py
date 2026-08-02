"""Eval-driven training loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError, Module
from agentnet.evaluation import EvaluationResult, Objective, aggregate_evaluation_results
from agentnet.mcp._security import validate_safe_metadata
from agentnet.training.budget import Budget
from agentnet.training.checkpoints import TrainingCheckpoint
from agentnet.training.datasets import Dataset, TrainingExample
from agentnet.training.history import TrainingHistory, TrainingStep
from agentnet.training.progress import TrainingProgressCallback, TrainingProgressEvent

CandidateScorer = Callable[[Module, EvaluationResult, TrainingHistory], float]


@dataclass(slots=True)
class TrainingCandidateResult:
    """Evaluation result for one live training candidate."""

    candidate: Module
    score: float
    objective_result: EvaluationResult
    history: TrainingHistory
    checkpoint: TrainingCheckpoint

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Module):
            raise AgentNetConfigurationError(
                "TrainingCandidateResult candidate must be a Module"
            )
        if not isinstance(self.objective_result, EvaluationResult):
            raise AgentNetConfigurationError(
                "TrainingCandidateResult objective_result must be an EvaluationResult"
            )
        if not isinstance(self.history, TrainingHistory):
            raise AgentNetConfigurationError(
                "TrainingCandidateResult history must be a TrainingHistory"
            )
        if not isinstance(self.checkpoint, TrainingCheckpoint):
            raise AgentNetConfigurationError(
                "TrainingCandidateResult checkpoint must be a TrainingCheckpoint"
            )
        self.score = float(self.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_descriptor": _candidate_descriptor(self.candidate),
            "checkpoint": self.checkpoint.to_dict(),
            "history": self.history.to_dict(),
            "objective_result": self.objective_result.to_dict(),
            "score": self.score,
        }


@dataclass(slots=True)
class TrainingResult:
    """Result of fitting a network against a dataset."""

    net: Module
    score: float
    objective_result: EvaluationResult
    history: TrainingHistory
    checkpoints: tuple[TrainingCheckpoint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_results: tuple[TrainingCandidateResult, ...] = ()

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        metadata["selection_policy"] = "first"
        validate_safe_metadata(metadata, label="TrainingResult")
        candidate_results = tuple(self.candidate_results)
        for candidate_result in candidate_results:
            if not isinstance(candidate_result, TrainingCandidateResult):
                raise AgentNetConfigurationError(
                    "TrainingResult candidate_results must contain "
                    "TrainingCandidateResult instances"
                )

        if candidate_results:
            selected = max(candidate_results, key=lambda result: result.score)
            self.net = selected.candidate
            self.score = selected.score
            self.objective_result = selected.objective_result
            self.history = selected.history
            self.checkpoints = tuple(
                result.checkpoint for result in candidate_results
            )
        else:
            self.score = float(self.score)
            self.checkpoints = tuple(self.checkpoints)

        self.metadata = metadata
        self.candidate_results = candidate_results

    @property
    def selected_candidate_result(self) -> TrainingCandidateResult | None:
        """Return the first highest-scoring candidate result, when available."""

        if not self.candidate_results:
            return None
        return max(self.candidate_results, key=lambda result: result.score)

    @property
    def tied_candidates(self) -> tuple[TrainingCandidateResult, ...]:
        """Return all candidates tied at the selected score in evaluation order."""

        if not self.candidate_results:
            return ()
        return tuple(
            result for result in self.candidate_results if result.score == self.score
        )

    @property
    def is_tied(self) -> bool:
        """Whether multiple candidates share the selected score."""

        return len(self.tied_candidates) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_results": [
                candidate_result.to_dict()
                for candidate_result in self.candidate_results
            ],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "history": self.history.to_dict(),
            "is_tied": self.is_tied,
            "metadata": self.metadata.copy(),
            "net_descriptor": _candidate_descriptor(self.net),
            "objective_result": self.objective_result.to_dict(),
            "score": self.score,
            "tied_candidates": [
                _candidate_descriptor(candidate_result.candidate)
                for candidate_result in self.tied_candidates
            ],
        }


class Trainer:
    """Evaluate and select trainable network candidates against an objective."""

    def __init__(
        self,
        *,
        net: Module,
        objective: Objective,
        optimizer: object | None = None,
        budget: Budget | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(net, Module):
            raise AgentNetConfigurationError("Trainer net must be a Module")
        if not isinstance(objective, Objective):
            raise AgentNetConfigurationError("Trainer objective must be an Objective")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="Trainer")

        self.net = net
        self.objective = objective
        self.optimizer = optimizer
        self.budget = budget if budget is not None else Budget()
        self.metadata = metadata_copy

    def fit(
        self,
        dataset: Dataset | Iterable[TrainingExample | Mapping[str, Any]],
        *,
        epochs: int = 1,
        candidates: Iterable[Module] | None = None,
        candidate_scorer: CandidateScorer | None = None,
        progress_callback: TrainingProgressCallback | None = None,
    ) -> TrainingResult:
        training_dataset = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        if not training_dataset:
            raise AgentNetValidationError("Trainer requires at least one training example")
        if epochs < 1:
            raise AgentNetConfigurationError("Trainer epochs must be at least 1")

        candidate_tuple = tuple(candidates or (self.net,))
        if not candidate_tuple:
            raise AgentNetValidationError("Trainer requires at least one candidate network")
        for candidate in candidate_tuple:
            if not isinstance(candidate, Module):
                raise AgentNetConfigurationError("Trainer candidates must be Module instances")

        total_examples = len(training_dataset) * epochs * len(candidate_tuple)
        if not self.budget.can_run(
            epoch=epochs,
            examples=total_examples,
            trials=len(candidate_tuple),
        ):
            raise AgentNetValidationError("Training budget would be exceeded")

        example_count = len(training_dataset) * epochs
        _emit_progress(
            progress_callback,
            TrainingProgressEvent(
                "training.started",
                candidate_count=len(candidate_tuple),
                example_count=example_count,
                epochs=epochs,
            ),
        )

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_index, candidate in enumerate(candidate_tuple):
            candidate_descriptor = _candidate_descriptor(candidate)
            _emit_progress(
                progress_callback,
                TrainingProgressEvent(
                    "candidate.started",
                    candidate_count=len(candidate_tuple),
                    example_count=example_count,
                    epochs=epochs,
                    candidate_index=candidate_index,
                    candidate_descriptor=candidate_descriptor,
                ),
            )
            history = TrainingHistory(metadata={"dataset": training_dataset.name})
            objective_result = self._evaluate_candidate(
                candidate,
                training_dataset,
                epochs=epochs,
                history=history,
                candidate_index=candidate_index,
                candidate_count=len(candidate_tuple),
                progress_callback=progress_callback,
            )
            score = (
                objective_result.score
                if candidate_scorer is None
                else float(candidate_scorer(candidate, objective_result, history))
            )
            checkpoint = TrainingCheckpoint(
                epoch=epochs,
                step=len(history.steps),
                score=score,
                objective_result=objective_result,
                history=history,
                candidate_descriptor=_candidate_descriptor(candidate),
                metadata={"dataset": training_dataset.name},
            )
            candidate_result = TrainingCandidateResult(
                candidate=candidate,
                score=score,
                objective_result=objective_result,
                history=history,
                checkpoint=checkpoint,
            )
            candidate_results.append(candidate_result)
            _emit_progress(
                progress_callback,
                TrainingProgressEvent(
                    "candidate.completed",
                    candidate_count=len(candidate_tuple),
                    example_count=example_count,
                    epochs=epochs,
                    candidate_index=candidate_index,
                    candidate_descriptor=candidate_descriptor,
                    score=score,
                    passed=objective_result.passed,
                ),
            )

        self.budget.record(examples=total_examples, trials=len(candidate_tuple))
        if not candidate_results:
            raise AgentNetValidationError("No training candidate was evaluated")
        selected = max(candidate_results, key=lambda result: result.score)
        result = TrainingResult(
            net=selected.candidate,
            score=selected.score,
            objective_result=selected.objective_result,
            history=selected.history,
            checkpoints=tuple(
                candidate_result.checkpoint
                for candidate_result in candidate_results
            ),
            metadata={
                **self.metadata,
                "dataset": training_dataset.name,
                "epochs": epochs,
                "evaluated_candidates": len(candidate_tuple),
                "optimizer": self.optimizer.__class__.__name__
                if self.optimizer is not None
                else None,
                "selection_policy": "first",
            },
            candidate_results=tuple(candidate_results),
        )
        selected_index = candidate_results.index(selected)
        _emit_progress(
            progress_callback,
            TrainingProgressEvent(
                "training.completed",
                candidate_count=len(candidate_tuple),
                example_count=example_count,
                epochs=epochs,
                candidate_index=selected_index,
                candidate_descriptor=_candidate_descriptor(selected.candidate),
                score=selected.score,
                passed=selected.objective_result.passed,
            ),
        )
        return result

    def _evaluate_candidate(
        self,
        candidate: Module,
        dataset: Dataset,
        *,
        epochs: int,
        history: TrainingHistory,
        candidate_index: int,
        candidate_count: int,
        progress_callback: TrainingProgressCallback | None,
    ) -> EvaluationResult:
        results: list[EvaluationResult] = []
        example_count = len(dataset) * epochs
        candidate_descriptor = _candidate_descriptor(candidate)
        for epoch in range(1, epochs + 1):
            for index, example in enumerate(dataset):
                example_index = (epoch - 1) * len(dataset) + index
                event_fields = {
                    "candidate_count": candidate_count,
                    "example_count": example_count,
                    "epochs": epochs,
                    "candidate_index": candidate_index,
                    "example_index": example_index,
                    "epoch": epoch,
                    "example_id": example.id,
                    "candidate_descriptor": candidate_descriptor,
                }
                _emit_progress(
                    progress_callback,
                    TrainingProgressEvent("example.started", **event_fields),
                )
                output = candidate.run(example.input)
                evaluation = self.objective.evaluate(
                    output,
                    context={
                        "epoch": epoch,
                        "example": example,
                        "example_index": index,
                        "expected_output": example.expected_output,
                    },
                )
                results.append(evaluation)
                history.add(
                    TrainingStep(
                        epoch=epoch,
                        example_id=example.id,
                        score=evaluation.score,
                        passed=evaluation.passed,
                        metrics=evaluation.metrics,
                        metadata={
                            "failure_count": len(evaluation.failures),
                        },
                    )
                )
                _emit_progress(
                    progress_callback,
                    TrainingProgressEvent(
                        "example.completed",
                        **event_fields,
                        score=evaluation.score,
                        passed=evaluation.passed,
                    ),
                )

        aggregate = aggregate_evaluation_results(results)
        metrics = dict(aggregate.metrics)
        metrics["training.example_count"] = float(len(results))
        return EvaluationResult(
            score=aggregate.score,
            passed=aggregate.passed,
            failures=aggregate.failures,
            metrics=metrics,
            metadata={
                **aggregate.metadata,
                "epochs": epochs,
                "training_examples": len(results),
            },
        )


def _candidate_descriptor(candidate: Module) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "type": candidate.__class__.__name__,
    }


def _emit_progress(
    callback: TrainingProgressCallback | None,
    event: TrainingProgressEvent,
) -> None:
    if callback is not None:
        callback(event)
