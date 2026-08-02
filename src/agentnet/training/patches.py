"""Training patch generation and rollback."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Self
from uuid import uuid4

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError
from agentnet.mcp._security import validate_safe_metadata

_PATCH_KINDS = {
    "fallback_order",
    "handoff",
    "prompt",
    "retry_policy",
    "tool_policy",
    "topology",
}


@dataclass(frozen=True, slots=True, init=False)
class TrainingPatch:
    """Serializable patch for trainable AgentNet state."""

    id: str
    target: str
    field: str
    kind: str
    old_value: Any
    new_value: Any
    rationale: str
    before_score: float | None
    after_score: float | None
    trace_evidence: tuple[str, ...]
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __init__(
        self,
        *,
        target: str,
        field: str,
        kind: str,
        old_value: Any,
        new_value: Any,
        rationale: str,
        id: str | None = None,
        before_score: float | None = None,
        after_score: float | None = None,
        trace_evidence: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        patch_id = id or str(uuid4())
        if not patch_id:
            raise AgentNetConfigurationError("TrainingPatch id cannot be empty")
        if not target:
            raise AgentNetConfigurationError("TrainingPatch target cannot be empty")
        if not field:
            raise AgentNetConfigurationError("TrainingPatch field cannot be empty")
        if kind not in _PATCH_KINDS:
            raise AgentNetConfigurationError(
                f"TrainingPatch kind must be one of: {', '.join(sorted(_PATCH_KINDS))}"
            )
        if not rationale:
            raise AgentNetConfigurationError("TrainingPatch rationale cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="TrainingPatch")

        object.__setattr__(self, "id", patch_id)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "old_value", deepcopy(old_value))
        object.__setattr__(self, "new_value", deepcopy(new_value))
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(
            self,
            "before_score",
            None if before_score is None else float(before_score),
        )
        object.__setattr__(
            self,
            "after_score",
            None if after_score is None else float(after_score),
        )
        object.__setattr__(
            self,
            "trace_evidence",
            tuple(str(item) for item in trace_evidence),
        )
        object.__setattr__(self, "metadata", metadata_copy)

    @property
    def diff(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "new_value": deepcopy(self.new_value),
            "old_value": deepcopy(self.old_value),
        }

    def apply(self, targets: Mapping[str, object]) -> None:
        _write_field(_resolve_target(targets, self.target), self.field, self.new_value)

    def rollback(self, targets: Mapping[str, object]) -> None:
        _write_field(_resolve_target(targets, self.target), self.field, self.old_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "after_score": self.after_score,
            "before_score": self.before_score,
            "diff": self.diff,
            "field": self.field,
            "id": self.id,
            "kind": self.kind,
            "metadata": self.metadata.copy(),
            "new_value": deepcopy(self.new_value),
            "old_value": deepcopy(self.old_value),
            "rationale": self.rationale,
            "target": self.target,
            "trace_evidence": list(self.trace_evidence),
        }

    @classmethod
    def from_dict(cls, patch: Mapping[str, Any]) -> Self:
        return cls(
            id=str(patch["id"]),
            target=str(patch["target"]),
            field=str(patch["field"]),
            kind=str(patch["kind"]),
            old_value=patch.get("old_value"),
            new_value=patch.get("new_value"),
            rationale=str(patch["rationale"]),
            before_score=(
                None if patch.get("before_score") is None else float(patch["before_score"])
            ),
            after_score=(
                None if patch.get("after_score") is None else float(patch["after_score"])
            ),
            trace_evidence=tuple(str(item) for item in patch.get("trace_evidence", ())),
            metadata=dict(patch.get("metadata", {})),
        )


def generate_training_patch(
    *,
    target: object,
    target_name: str,
    field: str,
    kind: str,
    new_value: Any,
    rationale: str,
    before_score: float | None = None,
    after_score: float | None = None,
    trace_evidence: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> TrainingPatch:
    """Generate a reversible patch from the target's current field value."""

    old_value = _read_field(target, field)
    return TrainingPatch(
        target=target_name,
        field=field,
        kind=kind,
        old_value=old_value,
        new_value=new_value,
        rationale=rationale,
        before_score=before_score,
        after_score=after_score,
        trace_evidence=trace_evidence,
        metadata=metadata,
    )


def _resolve_target(targets: Mapping[str, object], target_name: str) -> object:
    if target_name not in targets:
        raise AgentNetValidationError(f"Patch target {target_name!r} was not provided")
    return targets[target_name]


def _read_field(target: object, field: str) -> Any:
    if isinstance(target, Mapping):
        if field not in target:
            raise AgentNetValidationError(f"Patch field {field!r} is missing")
        return deepcopy(target[field])
    if not hasattr(target, field):
        raise AgentNetValidationError(f"Patch field {field!r} is missing")
    return deepcopy(getattr(target, field))


def _write_field(target: object, field: str, value: Any) -> None:
    if isinstance(target, MutableMapping):
        target[field] = deepcopy(value)
        return
    if isinstance(target, Mapping):
        raise AgentNetValidationError("Patch target mapping is not mutable")
    if not hasattr(target, field):
        raise AgentNetValidationError(f"Patch field {field!r} is missing")
    setattr(target, field, deepcopy(value))
