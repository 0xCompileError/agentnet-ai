"""Serializable training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError
from agentnet.evaluation import EvaluationResult
from agentnet.mcp._security import (
    validate_descriptor_payload_no_secrets,
    validate_safe_metadata,
)
from agentnet.training.history import TrainingHistory


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Descriptor-only snapshot of training progress."""

    epoch: int
    step: int
    score: float
    objective_result: EvaluationResult
    history: TrainingHistory
    candidate_descriptor: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise AgentNetConfigurationError("TrainingCheckpoint epoch must be at least 1")
        if self.step < 0:
            raise AgentNetConfigurationError("TrainingCheckpoint step cannot be negative")
        candidate_descriptor = dict(self.candidate_descriptor)
        metadata = dict(self.metadata)
        validate_descriptor_payload_no_secrets(
            candidate_descriptor,
            label="TrainingCheckpoint candidate_descriptor",
        )
        validate_safe_metadata(metadata, label="TrainingCheckpoint")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "candidate_descriptor", candidate_descriptor)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_descriptor": dict(self.candidate_descriptor),
            "epoch": self.epoch,
            "history": self.history.to_dict(),
            "metadata": dict(self.metadata),
            "objective_result": self.objective_result.to_dict(),
            "score": self.score,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, checkpoint: Mapping[str, Any]) -> Self:
        return cls(
            epoch=int(checkpoint["epoch"]),
            step=int(checkpoint["step"]),
            score=float(checkpoint["score"]),
            objective_result=EvaluationResult.from_dict(
                dict(checkpoint["objective_result"])
            ),
            history=TrainingHistory.from_dict(dict(checkpoint["history"])),
            candidate_descriptor=dict(checkpoint.get("candidate_descriptor", {})),
            metadata=dict(checkpoint.get("metadata", {})),
        )
