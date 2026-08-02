"""Training history records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError, AgentNetStateError
from agentnet.mcp._security import validate_safe_metadata


@dataclass(frozen=True, slots=True, init=False)
class TrainingStep:
    """Result of evaluating one example during training."""

    epoch: int
    example_id: str | None
    score: float
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        epoch: int,
        example_id: str | None,
        score: float,
        passed: bool,
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if epoch < 1:
            raise AgentNetConfigurationError("TrainingStep epoch must be at least 1")
        if example_id is not None and not example_id:
            raise AgentNetConfigurationError("TrainingStep example_id cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="TrainingStep")

        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "score", float(score))
        object.__setattr__(self, "passed", bool(passed))
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in dict(metrics or {}).items()},
        )
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "example_id": self.example_id,
            "metadata": self.metadata.copy(),
            "metrics": self.metrics.copy(),
            "passed": self.passed,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, step: Mapping[str, Any]) -> Self:
        return cls(
            epoch=int(step["epoch"]),
            example_id=None if step.get("example_id") is None else str(step["example_id"]),
            score=float(step["score"]),
            passed=bool(step["passed"]),
            metrics={
                str(key): float(value)
                for key, value in dict(step.get("metrics", {})).items()
            },
            metadata=dict(step.get("metadata", {})),
        )


class TrainingHistory:
    """Mutable append-only record of training evaluation steps."""

    def __init__(
        self,
        steps: Iterable[TrainingStep | Mapping[str, Any]] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="TrainingHistory")
        self._steps = [_coerce_step(step) for step in steps or ()]
        self.metadata = metadata_copy

    @property
    def steps(self) -> tuple[TrainingStep, ...]:
        return tuple(self._steps)

    def add(self, step: TrainingStep) -> None:
        if not isinstance(step, TrainingStep):
            raise AgentNetConfigurationError("TrainingHistory add requires TrainingStep")
        self._steps.append(step)

    @property
    def best_step(self) -> TrainingStep:
        if not self._steps:
            raise AgentNetStateError("TrainingHistory is empty")
        return max(self._steps, key=lambda step: step.score)

    @property
    def best_score(self) -> float | None:
        if not self._steps:
            return None
        return self.best_step.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.copy(),
            "steps": [step.to_dict() for step in self._steps],
        }

    @classmethod
    def from_dict(cls, history: Mapping[str, Any]) -> Self:
        return cls(
            [TrainingStep.from_dict(dict(step)) for step in history.get("steps", ())],
            metadata=dict(history.get("metadata", {})),
        )


def _coerce_step(step: TrainingStep | Mapping[str, Any]) -> TrainingStep:
    if isinstance(step, TrainingStep):
        return step
    return TrainingStep.from_dict(step)
