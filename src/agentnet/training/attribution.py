"""Attribution records for training patches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from agentnet.evaluation import EvaluationResult
from agentnet.mcp._security import validate_safe_metadata
from agentnet.training.patches import TrainingPatch


@dataclass(frozen=True, slots=True)
class AttributionRecord:
    """Score and metric deltas attributed to one training patch."""

    patch_id: str
    score_delta: float
    metric_deltas: Mapping[str, float] = field(default_factory=dict)
    before_score: float = 0.0
    after_score: float = 0.0
    evidence: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="AttributionRecord")
        object.__setattr__(self, "score_delta", float(self.score_delta))
        object.__setattr__(
            self,
            "metric_deltas",
            {str(key): float(value) for key, value in self.metric_deltas.items()},
        )
        object.__setattr__(self, "before_score", float(self.before_score))
        object.__setattr__(self, "after_score", float(self.after_score))
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "after_score": self.after_score,
            "before_score": self.before_score,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
            "metric_deltas": dict(self.metric_deltas),
            "patch_id": self.patch_id,
            "score_delta": self.score_delta,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        return cls(
            patch_id=str(record["patch_id"]),
            score_delta=float(record["score_delta"]),
            metric_deltas={
                str(key): float(value)
                for key, value in dict(record.get("metric_deltas", {})).items()
            },
            before_score=float(record.get("before_score", 0.0)),
            after_score=float(record.get("after_score", 0.0)),
            evidence=tuple(str(item) for item in record.get("evidence", ())),
            metadata=dict(record.get("metadata", {})),
        )


class AttributionEngine:
    """Compute objective-score deltas for accepted or rejected patches."""

    def attribute(
        self,
        *,
        patch: TrainingPatch,
        before: EvaluationResult,
        after: EvaluationResult,
        evidence: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AttributionRecord:
        metric_keys = set(before.metrics) | set(after.metrics)
        metric_deltas = {
            key: after.metrics.get(key, 0.0) - before.metrics.get(key, 0.0)
            for key in sorted(metric_keys)
        }
        return AttributionRecord(
            patch_id=patch.id,
            score_delta=after.score - before.score,
            metric_deltas=metric_deltas,
            before_score=before.score,
            after_score=after.score,
            evidence=tuple(evidence or ()),
            metadata={
                **dict(metadata or {}),
                "patch_kind": patch.kind,
                "patch_target": patch.target,
            },
        )
