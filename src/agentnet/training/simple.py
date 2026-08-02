"""Beginner-facing end-to-end training API."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from agentnet.constraints import validate_training_constraints
from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetValidationError,
    Module,
)
from agentnet.evaluation import (
    EvaluationResult,
    ExpectedOutputObjective,
    Objective,
    aggregate_evaluation_results,
)
from agentnet.graphs import validate_graph
from agentnet.optimizers import TopologyCandidate, TopologyOptimizer
from agentnet.training.automatic import (
    AutoOptimizer,
    ExplicitCandidates,
    TrainingCandidate,
    estimate_llm_calls,
    module_complexity,
)
from agentnet.training.budget import Budget
from agentnet.training.datasets import Dataset, TrainingExample
from agentnet.training.history import TrainingHistory, TrainingStep

TrainingProgress = Callable[["TrainingTrialEvent"], None]
ValidationData = Dataset | tuple[Iterable[Any], Iterable[Any]]


@dataclass(frozen=True, slots=True)
class TrainingTrial:
    """One safe, inspectable candidate evaluation in a simple training run."""

    index: int
    stage: str
    net: Module
    status: str
    train_result: EvaluationResult | None = None
    validation_result: EvaluationResult | None = None
    train_history: TrainingHistory | None = None
    validation_history: TrainingHistory | None = None
    changes: tuple[str, ...] = ()
    complexity: tuple[int, int, int] = (1, 0, 1)
    estimated_llm_calls: int = 0
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise AgentNetConfigurationError("TrainingTrial index cannot be negative")
        if self.status not in {"completed", "failed"}:
            raise AgentNetConfigurationError(
                "TrainingTrial status must be 'completed' or 'failed'"
            )
        if not isinstance(self.net, Module):
            raise AgentNetConfigurationError("TrainingTrial net must be a Module")
        object.__setattr__(self, "changes", tuple(self.changes))
        object.__setattr__(self, "complexity", tuple(self.complexity))

    @property
    def train_score(self) -> float | None:
        return None if self.train_result is None else self.train_result.score

    @property
    def validation_score(self) -> float | None:
        return None if self.validation_result is None else self.validation_result.score

    @property
    def selection_score(self) -> float | None:
        return self.validation_score if self.validation_score is not None else self.train_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": list(self.changes),
            "complexity": {
                "branches": self.complexity[1],
                "depth": self.complexity[2],
                "nodes": self.complexity[0],
            },
            "error_type": self.error_type,
            "estimated_llm_calls": self.estimated_llm_calls,
            "index": self.index,
            "net_descriptor": {
                "name": self.net.name,
                "type": self.net.__class__.__name__,
            },
            "stage": self.stage,
            "status": self.status,
            "train_result": (
                None if self.train_result is None else _result_summary(self.train_result)
            ),
            "validation_result": (
                None
                if self.validation_result is None
                else _result_summary(self.validation_result)
            ),
        }


@dataclass(frozen=True, slots=True)
class TrainingTrialEvent:
    """Prompt-free progress event emitted after a simple-training trial."""

    index: int
    stage: str
    status: str
    net_descriptor: Mapping[str, str]
    changes: tuple[str, ...]
    train_score: float | None
    validation_score: float | None
    error_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "net_descriptor", dict(self.net_descriptor))
        object.__setattr__(self, "changes", tuple(self.changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": list(self.changes),
            "error_type": self.error_type,
            "index": self.index,
            "net_descriptor": dict(self.net_descriptor),
            "stage": self.stage,
            "status": self.status,
            "train_score": self.train_score,
            "validation_score": self.validation_score,
        }


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Training provenance attached to a runnable fitted network."""

    trials: tuple[TrainingTrial, ...]
    best_trial_index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    optimizer: str
    budget: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trials = tuple(self.trials)
        if not trials:
            raise AgentNetConfigurationError("TrainingReport requires at least one trial")
        if not 0 <= self.best_trial_index < len(trials):
            raise AgentNetConfigurationError("TrainingReport best_trial_index is invalid")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "train_indices", tuple(self.train_indices))
        object.__setattr__(self, "validation_indices", tuple(self.validation_indices))
        object.__setattr__(self, "budget", dict(self.budget))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def best_trial(self) -> TrainingTrial:
        return self.trials[self.best_trial_index]

    @property
    def train_score(self) -> float:
        return float(self.best_trial.train_score or 0.0)

    @property
    def validation_score(self) -> float | None:
        return self.best_trial.validation_score

    @property
    def best_score(self) -> float:
        return float(self.best_trial.selection_score or 0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "best_net": self.best_trial.net.name,
            "best_score": self.best_score,
            "changes": list(self.best_trial.changes),
            "optimizer": self.optimizer,
            "train_score": self.train_score,
            "trial_count": len(self.trials),
            "validation_score": self.validation_score,
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_trial_index": self.best_trial_index,
            "budget": dict(self.budget),
            "metadata": dict(self.metadata),
            "optimizer": self.optimizer,
            "split": {
                "train_indices": list(self.train_indices),
                "validation_indices": list(self.validation_indices),
            },
            "trials": [trial.to_dict() for trial in self.trials],
            "warnings": list(self.warnings),
        }


class FittedAgentNet(Module):
    """Runnable network returned by :func:`agentnet.train`."""

    def __init__(self, net: Module, training: TrainingReport) -> None:
        if not isinstance(net, Module):
            raise AgentNetConfigurationError("FittedAgentNet net must be a Module")
        super().__init__(net.name)
        self.net = net
        self.training = training

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        return await self.net.arun(input, context)

    async def aevaluate(
        self,
        X: Dataset | Iterable[Any],
        y: Iterable[Any] | None = None,
        *,
        objective: Objective | None = None,
    ) -> EvaluationResult:
        dataset = _coerce_xy(X, y, name="evaluation")
        resolved_objective = _resolve_objective(dataset, objective)
        result, _ = await _evaluate_dataset(
            self.net,
            dataset,
            resolved_objective,
            epochs=1,
        )
        return result

    def evaluate(
        self,
        X: Dataset | Iterable[Any],
        y: Iterable[Any] | None = None,
        *,
        objective: Objective | None = None,
    ) -> EvaluationResult:
        return anyio.run(partial(self.aevaluate, X, y, objective=objective))

    def save(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        overwrite: bool = True,
    ) -> Any:
        from agentnet.artifacts import save

        return save(
            self.net,
            path,
            name=name,
            training_history=self.training.best_trial.train_history,
            metadata={
                "optimizer": self.training.optimizer,
                "training_score": self.training.best_score,
            },
            overwrite=overwrite,
        )

    def state_dict(self) -> dict[str, Any]:
        return self.net.state_dict()


def train(
    net: Module,
    X: Dataset | Iterable[Any],
    y: Iterable[Any] | None = None,
    *,
    validation_data: ValidationData | None = None,
    optimize: str | bool | AutoOptimizer | ExplicitCandidates | TopologyOptimizer = "auto",
    objective: Objective | None = None,
    budget: Budget | None = None,
    optimizer_llm: Any | None = None,
    random_state: int = 0,
    progress: TrainingProgress | None = None,
    epochs: int = 1,
) -> FittedAgentNet:
    """Train a network from ordinary inputs and expected outputs."""

    return anyio.run(
        partial(
            atrain,
            net,
            X,
            y,
            validation_data=validation_data,
            optimize=optimize,
            objective=objective,
            budget=budget,
            optimizer_llm=optimizer_llm,
            random_state=random_state,
            progress=progress,
            epochs=epochs,
        )
    )


async def atrain(
    net: Module,
    X: Dataset | Iterable[Any],
    y: Iterable[Any] | None = None,
    *,
    validation_data: ValidationData | None = None,
    optimize: str | bool | AutoOptimizer | ExplicitCandidates | TopologyOptimizer = "auto",
    objective: Objective | None = None,
    budget: Budget | None = None,
    optimizer_llm: Any | None = None,
    random_state: int = 0,
    progress: TrainingProgress | None = None,
    epochs: int = 1,
) -> FittedAgentNet:
    """Asynchronously train a network from ordinary inputs and expected outputs."""

    if not isinstance(net, Module):
        raise AgentNetConfigurationError("train requires a Module network")
    if epochs < 1:
        raise AgentNetConfigurationError("train epochs must be at least 1")
    complete_dataset = _coerce_xy(X, y, name="training")
    trainset, validationset, train_indices, validation_indices, warnings = _split_data(
        complete_dataset,
        validation_data=validation_data,
        random_state=random_state,
    )
    resolved_objective = _resolve_objective(complete_dataset, objective)
    resolved_budget = budget or Budget(max_trials=10, max_llm_calls=200)
    resolved_optimizer = _resolve_optimizer(optimize, optimizer_llm=optimizer_llm)
    optimizer_name = (
        "none"
        if resolved_optimizer is None
        else resolved_optimizer.__class__.__name__
    )

    trials: list[TrainingTrial] = []
    baseline = await _run_candidate(
        TrainingCandidate(net, "baseline"),
        index=0,
        trainset=trainset,
        validationset=validationset,
        objective=resolved_objective,
        epochs=epochs,
        budget=resolved_budget,
        required=True,
    )
    if baseline is None:  # pragma: no cover - required candidates raise instead
        raise AgentNetValidationError("The baseline network could not be evaluated")
    trials.append(baseline)
    _notify(progress, baseline)
    current = baseline

    if isinstance(resolved_optimizer, AutoOptimizer):
        current = await _run_auto_stages(
            resolved_optimizer,
            current=current,
            trials=trials,
            trainset=trainset,
            validationset=validationset,
            objective=resolved_objective,
            epochs=epochs,
            budget=resolved_budget,
            warnings=warnings,
            progress=progress,
        )
    elif isinstance(resolved_optimizer, ExplicitCandidates):
        candidates = tuple(
            candidate
            for candidate in resolved_optimizer.generate(net)
            if candidate.net is not net
        )
        current = await _run_stage_candidates(
            candidates,
            current=current,
            trials=trials,
            trainset=trainset,
            validationset=validationset,
            objective=resolved_objective,
            epochs=epochs,
            budget=resolved_budget,
            warnings=warnings,
            progress=progress,
        )
    elif isinstance(resolved_optimizer, TopologyOptimizer):
        current = await _run_stage_candidates(
            _topology_candidates(resolved_optimizer, net),
            current=current,
            trials=trials,
            trainset=trainset,
            validationset=validationset,
            objective=resolved_objective,
            epochs=epochs,
            budget=resolved_budget,
            warnings=warnings,
            progress=progress,
        )

    completed = [trial for trial in trials if trial.status == "completed"]
    if not completed:
        raise AgentNetValidationError("No training candidate completed successfully")
    best = max(completed, key=_selection_key)
    report = TrainingReport(
        trials=tuple(trials),
        best_trial_index=trials.index(best),
        train_indices=train_indices,
        validation_indices=validation_indices,
        optimizer=optimizer_name,
        budget=resolved_budget.to_dict(),
        warnings=tuple(warnings),
        metadata={
            "epochs": epochs,
            "random_state": random_state,
            "selection_policy": "validation_train_simplicity_order",
        },
    )
    return FittedAgentNet(best.net, report)


async def _run_auto_stages(
    optimizer: AutoOptimizer,
    *,
    current: TrainingTrial,
    trials: list[TrainingTrial],
    trainset: Dataset,
    validationset: Dataset | None,
    objective: Objective,
    epochs: int,
    budget: Budget,
    warnings: list[str],
    progress: TrainingProgress | None,
) -> TrainingTrial:
    failure_examples = _failure_examples(current, trainset)
    stages: tuple[tuple[str, Any], ...] = (
        ("prompt", optimizer.prompt_candidates),
        ("topology", optimizer.topology_candidates),
    )
    for stage, generator in stages:
        proposal_llm = optimizer.proposal_llm(current.net)
        if proposal_llm is None:
            warnings.append(f"{stage} optimization skipped: no proposal LLM available")
            continue
        if not budget.can_run(llm_calls=1):
            warnings.append(f"{stage} optimization skipped: LLM-call budget exhausted")
            continue
        try:
            candidates = await generator(
                current.net,
                failure_examples=failure_examples,
            )
        except Exception as exc:
            warnings.append(
                f"{stage} proposal failed safely ({type(exc).__name__})"
            )
            continue
        budget.record(llm_calls=1)
        current = await _run_stage_candidates(
            candidates,
            current=current,
            trials=trials,
            trainset=trainset,
            validationset=validationset,
            objective=objective,
            epochs=epochs,
            budget=budget,
            warnings=warnings,
            progress=progress,
        )
        failure_examples = _failure_examples(current, trainset)

    current = await _run_stage_candidates(
        optimizer.policy_candidates(current.net),
        current=current,
        trials=trials,
        trainset=trainset,
        validationset=validationset,
        objective=objective,
        epochs=epochs,
        budget=budget,
        warnings=warnings,
        progress=progress,
    )
    return current


async def _run_stage_candidates(
    candidates: Sequence[TrainingCandidate],
    *,
    current: TrainingTrial,
    trials: list[TrainingTrial],
    trainset: Dataset,
    validationset: Dataset | None,
    objective: Objective,
    epochs: int,
    budget: Budget,
    warnings: list[str],
    progress: TrainingProgress | None,
) -> TrainingTrial:
    stage_trials: list[TrainingTrial] = []
    for candidate in candidates:
        trial = await _run_candidate(
            candidate,
            index=len(trials),
            trainset=trainset,
            validationset=validationset,
            objective=objective,
            epochs=epochs,
            budget=budget,
            required=False,
        )
        if trial is None:
            warnings.append(
                f"{candidate.stage} candidate skipped: training budget exhausted"
            )
            break
        trials.append(trial)
        stage_trials.append(trial)
        _notify(progress, trial)
    successful = [trial for trial in stage_trials if trial.status == "completed"]
    if not successful:
        return current
    stage_best = max(successful, key=_selection_key)
    return stage_best if _selection_key(stage_best) > _selection_key(current) else current


async def _run_candidate(
    candidate: TrainingCandidate,
    *,
    index: int,
    trainset: Dataset,
    validationset: Dataset | None,
    objective: Objective,
    epochs: int,
    budget: Budget,
    required: bool,
) -> TrainingTrial | None:
    example_count = (len(trainset) + len(validationset or ())) * epochs
    llm_calls = estimate_llm_calls(candidate.net, example_count)
    if not budget.can_run(
        epoch=epochs,
        examples=example_count,
        trials=1,
        llm_calls=llm_calls,
    ):
        if required:
            raise AgentNetValidationError(
                "Training baseline exceeds the configured budget; pass a larger Budget"
            )
        return None

    train_result: EvaluationResult | None = None
    train_history: TrainingHistory | None = None
    try:
        train_result, train_history = await _evaluate_dataset(
            candidate.net,
            trainset,
            objective,
            epochs=epochs,
        )
        validation_result: EvaluationResult | None = None
        validation_history: TrainingHistory | None = None
        if validationset is not None:
            validation_result, validation_history = await _evaluate_dataset(
                candidate.net,
                validationset,
                objective,
                epochs=epochs,
            )
        trial = TrainingTrial(
            index=index,
            stage=candidate.stage,
            net=candidate.net,
            status="completed",
            train_result=train_result,
            validation_result=validation_result,
            train_history=train_history,
            validation_history=validation_history,
            changes=candidate.changes,
            complexity=module_complexity(candidate.net),
            estimated_llm_calls=llm_calls,
        )
    except Exception as exc:
        trial = TrainingTrial(
            index=index,
            stage=candidate.stage,
            net=candidate.net,
            status="failed",
            train_result=train_result,
            train_history=train_history,
            changes=candidate.changes,
            complexity=module_complexity(candidate.net),
            estimated_llm_calls=llm_calls,
            error_type=type(exc).__name__,
        )
    budget.record(
        examples=example_count,
        trials=1,
        llm_calls=llm_calls,
    )
    return trial


async def _evaluate_dataset(
    net: Module,
    dataset: Dataset,
    objective: Objective,
    *,
    epochs: int,
) -> tuple[EvaluationResult, TrainingHistory]:
    history = TrainingHistory(metadata={"dataset": dataset.name})
    results: list[EvaluationResult] = []
    for epoch in range(1, epochs + 1):
        for index, example in enumerate(dataset):
            output = await net.arun(example.input)
            evaluation = objective.evaluate(
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
                    metadata={"failure_count": len(evaluation.failures)},
                )
            )
    aggregate = aggregate_evaluation_results(results)
    metrics = dict(aggregate.metrics)
    metrics["training.example_count"] = float(len(results))
    return (
        EvaluationResult(
            score=aggregate.score,
            passed=aggregate.passed,
            failures=aggregate.failures,
            metrics=metrics,
            metadata={
                **aggregate.metadata,
                "epochs": epochs,
                "training_examples": len(results),
            },
        ),
        history,
    )


def _coerce_xy(
    X: Dataset | Iterable[Any],
    y: Iterable[Any] | None,
    *,
    name: str,
) -> Dataset:
    if isinstance(X, Dataset):
        if y is not None:
            raise AgentNetConfigurationError(
                "y must be omitted when X is already an AgentNet Dataset"
            )
        if not X:
            raise AgentNetValidationError("Training data cannot be empty")
        return X
    if y is None:
        raise AgentNetConfigurationError(
            "Expected outputs are required: call train(net, X, y) or pass a Dataset"
        )
    inputs = tuple(X)
    expected = tuple(y)
    if not inputs:
        raise AgentNetValidationError("Training data cannot be empty")
    if len(inputs) != len(expected):
        raise AgentNetValidationError(
            f"X and y contain different numbers of examples ({len(inputs)} != {len(expected)})"
        )
    return Dataset(
        [
            TrainingExample(value, expected_output=expected[index], id=f"case-{index + 1}")
            for index, value in enumerate(inputs)
        ],
        name=name,
    )


def _split_data(
    dataset: Dataset,
    *,
    validation_data: ValidationData | None,
    random_state: int,
) -> tuple[Dataset, Dataset | None, tuple[int, ...], tuple[int, ...], list[str]]:
    warnings: list[str] = []
    all_indices = tuple(range(len(dataset)))
    if validation_data is not None:
        if isinstance(validation_data, Dataset):
            validation = _coerce_xy(validation_data, None, name="validation")
        else:
            validation = _coerce_xy(
                validation_data[0],
                validation_data[1],
                name="validation",
            )
        return dataset, validation, all_indices, (), warnings
    if len(dataset) < 5:
        warnings.append(
            "validation split skipped because fewer than five examples were provided"
        )
        return dataset, None, all_indices, (), warnings

    validation_count = max(1, round(len(dataset) * 0.2))
    validation_indices = _validation_indices(
        dataset,
        count=validation_count,
        random_state=random_state,
    )
    validation_set = set(validation_indices)
    train_indices = tuple(index for index in all_indices if index not in validation_set)
    return (
        _dataset_subset(dataset, train_indices, "train"),
        _dataset_subset(dataset, validation_indices, "validation"),
        train_indices,
        validation_indices,
        warnings,
    )


def _validation_indices(
    dataset: Dataset,
    *,
    count: int,
    random_state: int,
) -> tuple[int, ...]:
    rng = random.Random(random_state)
    groups: dict[Hashable, list[int]] = defaultdict(list)
    stratifiable = True
    for index, example in enumerate(dataset):
        label = example.expected_output
        try:
            hash(label)
        except TypeError:
            stratifiable = False
            break
        groups[label].append(index)
    if (
        stratifiable
        and len(groups) > 1
        and count >= len(groups)
        and all(len(indices) >= 2 for indices in groups.values())
    ):
        selected: list[int] = []
        remaining: list[int] = []
        for label in sorted(groups, key=str):
            indices = list(groups[label])
            rng.shuffle(indices)
            selected.append(indices.pop())
            remaining.extend(indices)
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
        return tuple(sorted(selected))
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return tuple(sorted(indices[:count]))


def _dataset_subset(dataset: Dataset, indices: Sequence[int], split: str) -> Dataset:
    metadata = dict(dataset.metadata)
    metadata["split"] = split
    return Dataset(
        [dataset[index] for index in indices],
        name=f"{dataset.name or 'dataset'}-{split}",
        metadata=metadata,
    )


def _resolve_objective(dataset: Dataset, objective: Objective | None) -> Objective:
    if objective is not None:
        if not isinstance(objective, Objective):
            raise AgentNetConfigurationError("objective must be an Objective")
        return objective
    missing = [
        example.id or str(index)
        for index, example in enumerate(dataset)
        if example.expected_output is None
    ]
    if missing:
        preview = ", ".join(missing[:3])
        raise AgentNetConfigurationError(
            "Cannot infer an objective because expected_output is missing for "
            f"example(s): {preview}; pass an explicit objective"
        )
    return ExpectedOutputObjective()


def _resolve_optimizer(
    optimize: str | bool | AutoOptimizer | ExplicitCandidates | TopologyOptimizer,
    *,
    optimizer_llm: Any | None,
) -> AutoOptimizer | ExplicitCandidates | TopologyOptimizer | None:
    if optimize is False or optimize == "none":
        return None
    if isinstance(optimize, AutoOptimizer):
        if optimizer_llm is not None:
            raise AgentNetConfigurationError(
                "Pass optimizer_llm to AutoOptimizer or train, not both"
            )
        return optimize
    if isinstance(optimize, ExplicitCandidates | TopologyOptimizer):
        if optimizer_llm is not None:
            raise AgentNetConfigurationError(
                "optimizer_llm is only valid for automatic optimization"
            )
        return optimize
    if optimize is True or optimize == "auto":
        return AutoOptimizer(optimizer_llm=optimizer_llm)
    if optimize == "prompt":
        return AutoOptimizer(
            optimizer_llm=optimizer_llm,
            optimize_topology=False,
            optimize_policies=False,
        )
    if optimize == "topology":
        return AutoOptimizer(
            optimizer_llm=optimizer_llm,
            optimize_prompts=False,
            optimize_policies=False,
        )
    if optimize == "policy":
        return AutoOptimizer(
            optimizer_llm=optimizer_llm,
            optimize_prompts=False,
            optimize_topology=False,
        )
    raise AgentNetConfigurationError(
        "optimize must be 'auto', 'prompt', 'topology', 'policy', 'none', "
        "or a supported optimizer"
    )


def _topology_candidates(
    optimizer: TopologyOptimizer,
    seed: Module,
) -> tuple[TrainingCandidate, ...]:
    candidates: list[TrainingCandidate] = []
    for raw_candidate in optimizer.mutation_engine.generate(seed, optimizer.search_space):
        candidate = (
            raw_candidate
            if isinstance(raw_candidate, TopologyCandidate)
            else TopologyCandidate(raw_candidate)
        )
        try:
            graph = validate_graph(candidate.module)
            if optimizer.search_space.violation(graph) is not None:
                continue
            validate_training_constraints(graph, optimizer.constraints)
        except AgentNetValidationError:
            continue
        mutation = candidate.mutation
        changes = (
            ()
            if mutation is None
            else (f"topology:{mutation.kind}",)
        )
        candidates.append(TrainingCandidate(candidate.module, "topology", changes))
    return tuple(candidates)


def _selection_key(trial: TrainingTrial) -> tuple[float, float, int, int, int, int]:
    if trial.status != "completed":
        return (float("-inf"), float("-inf"), 0, 0, 0, -trial.index)
    validation = trial.validation_score
    train_score = float(trial.train_score or 0.0)
    primary = train_score if validation is None else float(validation)
    nodes, branches, depth = trial.complexity
    return (primary, train_score, -nodes, -branches, -depth, -trial.index)


def _failure_examples(
    trial: TrainingTrial,
    dataset: Dataset,
) -> tuple[Mapping[str, Any], ...]:
    if trial.train_history is None:
        return ()
    examples: list[Mapping[str, Any]] = []
    for step, example in zip(trial.train_history.steps, dataset, strict=False):
        if not step.passed:
            examples.append(
                {
                    "expected_output": example.expected_output,
                    "id": example.id,
                    "input": example.input,
                }
            )
        if len(examples) == 5:
            break
    return tuple(examples)


def _notify(progress: TrainingProgress | None, trial: TrainingTrial) -> None:
    if progress is not None:
        progress(
            TrainingTrialEvent(
                index=trial.index,
                stage=trial.stage,
                status=trial.status,
                net_descriptor={
                    "name": trial.net.name,
                    "type": trial.net.__class__.__name__,
                },
                changes=trial.changes,
                train_score=trial.train_score,
                validation_score=trial.validation_score,
                error_type=trial.error_type,
            )
        )


def _result_summary(result: EvaluationResult) -> dict[str, Any]:
    return {
        "metrics": dict(result.metrics),
        "passed": result.passed,
        "score": result.score,
    }
