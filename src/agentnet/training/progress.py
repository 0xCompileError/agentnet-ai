"""Descriptor-safe training progress events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError
from agentnet.mcp._security import validate_descriptor_payload_no_secrets

_EVENT_TYPES = frozenset(
    {
        "candidate.completed",
        "candidate.started",
        "example.completed",
        "example.started",
        "training.completed",
        "training.started",
    }
)


@dataclass(frozen=True, slots=True)
class TrainingProgressEvent:
    """One prompt-free lifecycle event emitted by :class:`Trainer`."""

    event_type: str
    candidate_count: int
    example_count: int
    epochs: int
    candidate_index: int | None = None
    example_index: int | None = None
    epoch: int | None = None
    example_id: str | None = None
    candidate_descriptor: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None
    passed: bool | None = None

    def __post_init__(self) -> None:
        if self.event_type not in _EVENT_TYPES:
            raise AgentNetConfigurationError(
                "TrainingProgressEvent event_type must be a training lifecycle event"
            )
        if self.candidate_count < 1:
            raise AgentNetConfigurationError(
                "TrainingProgressEvent candidate_count must be at least 1"
            )
        if self.example_count < 1:
            raise AgentNetConfigurationError(
                "TrainingProgressEvent example_count must be at least 1"
            )
        if self.epochs < 1:
            raise AgentNetConfigurationError(
                "TrainingProgressEvent epochs must be at least 1"
            )
        if self.candidate_index is not None and not (
            0 <= self.candidate_index < self.candidate_count
        ):
            raise AgentNetConfigurationError(
                "TrainingProgressEvent candidate_index is out of range"
            )
        if self.example_index is not None and not (
            0 <= self.example_index < self.example_count
        ):
            raise AgentNetConfigurationError(
                "TrainingProgressEvent example_index is out of range"
            )
        if self.epoch is not None and not 1 <= self.epoch <= self.epochs:
            raise AgentNetConfigurationError(
                "TrainingProgressEvent epoch is out of range"
            )
        if self.example_id is not None and not self.example_id:
            raise AgentNetConfigurationError(
                "TrainingProgressEvent example_id cannot be empty"
            )

        candidate_descriptor = dict(self.candidate_descriptor)
        validate_descriptor_payload_no_secrets(
            candidate_descriptor,
            label="TrainingProgressEvent candidate_descriptor",
        )
        object.__setattr__(self, "candidate_descriptor", candidate_descriptor)
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        if self.passed is not None:
            object.__setattr__(self, "passed", bool(self.passed))

    @property
    def type(self) -> str:
        """Return the serialized event type."""

        return self.event_type

    @property
    def candidate_number(self) -> int | None:
        """Return the one-based candidate number for display."""

        if self.candidate_index is None:
            return None
        return self.candidate_index + 1

    @property
    def example_number(self) -> int | None:
        """Return the one-based example number within the candidate run."""

        if self.example_index is None:
            return None
        return self.example_index + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "candidate_descriptor": dict(self.candidate_descriptor),
            "candidate_index": self.candidate_index,
            "candidate_number": self.candidate_number,
            "epoch": self.epoch,
            "epochs": self.epochs,
            "example_count": self.example_count,
            "example_id": self.example_id,
            "example_index": self.example_index,
            "example_number": self.example_number,
            "passed": self.passed,
            "score": self.score,
            "type": self.event_type,
        }

    @classmethod
    def from_dict(cls, event: Mapping[str, Any]) -> Self:
        return cls(
            event_type=str(event["type"]),
            candidate_count=int(event["candidate_count"]),
            example_count=int(event["example_count"]),
            epochs=int(event["epochs"]),
            candidate_index=(
                None
                if event.get("candidate_index") is None
                else int(event["candidate_index"])
            ),
            example_index=(
                None
                if event.get("example_index") is None
                else int(event["example_index"])
            ),
            epoch=None if event.get("epoch") is None else int(event["epoch"]),
            example_id=(
                None if event.get("example_id") is None else str(event["example_id"])
            ),
            candidate_descriptor=dict(event.get("candidate_descriptor", {})),
            score=None if event.get("score") is None else float(event["score"]),
            passed=None if event.get("passed") is None else bool(event["passed"]),
        )


TrainingProgressCallback = Callable[[TrainingProgressEvent], None]
