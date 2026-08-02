"""Constraint base abstractions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError


class ConstraintKind(StrEnum):
    """Strictness level for a constraint."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class ConstraintDescriptor:
    """Serializable, non-executable constraint descriptor."""

    type: str
    name: str
    version: str = "1"
    kind: ConstraintKind = ConstraintKind.HARD
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    children: tuple[ConstraintDescriptor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _coerce_constraint_version(self.version))
        object.__setattr__(self, "kind", _coerce_constraint_kind(self.kind))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "children", tuple(self.children))

    def to_dict(self) -> dict[str, Any]:
        return {
            "children": [child.to_dict() for child in self.children],
            "description": self.description,
            "kind": self.kind.value,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "parameters": self.parameters.copy(),
            "type": self.type,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, descriptor: dict[str, Any]) -> Self:
        return cls(
            type=str(descriptor["type"]),
            name=str(descriptor["name"]),
            version=str(descriptor.get("version", "1")),
            kind=_coerce_constraint_kind(descriptor.get("kind", ConstraintKind.HARD)),
            description=descriptor.get("description"),
            metadata=dict(descriptor.get("metadata", {})),
            parameters=dict(descriptor.get("parameters", {})),
            children=tuple(
                cls.from_dict(child) for child in descriptor.get("children", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Result of evaluating one constraint."""

    constraint: str
    passed: bool
    kind: ConstraintKind = ConstraintKind.HARD
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_constraint_kind(self.kind))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_hard(self) -> bool:
        return self.kind is ConstraintKind.HARD

    @property
    def is_soft(self) -> bool:
        return self.kind is ConstraintKind.SOFT

    @property
    def blocks_candidate(self) -> bool:
        return not self.passed and self.is_hard

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint,
            "kind": self.kind.value,
            "message": self.message,
            "metadata": self.metadata.copy(),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, result: dict[str, Any]) -> Self:
        return cls(
            constraint=str(result["constraint"]),
            passed=bool(result["passed"]),
            kind=_coerce_constraint_kind(result.get("kind", ConstraintKind.HARD)),
            message=result.get("message"),
            metadata=dict(result.get("metadata", {})),
        )


class Constraint:
    """Base class for constraints that evaluate runtime or optimizer candidates."""

    def __init__(
        self,
        name: str,
        *,
        description: str | None = None,
        kind: ConstraintKind | str = ConstraintKind.HARD,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not name:
            raise AgentNetConfigurationError("Constraint name cannot be empty")
        self.name = name
        self.description = description
        self.kind = _coerce_constraint_kind(kind)
        self.metadata = dict(metadata or {})
        self.version = _coerce_constraint_version(version)

    @property
    def is_hard(self) -> bool:
        return self.kind is ConstraintKind.HARD

    @property
    def is_soft(self) -> bool:
        return self.kind is ConstraintKind.SOFT

    def __and__(self, other: Constraint) -> Constraint:
        from agentnet.constraints.composition import AndConstraint

        return AndConstraint(self, other)

    def __or__(self, other: Constraint) -> Constraint:
        from agentnet.constraints.composition import OrConstraint

        return OrConstraint(self, other)

    def __invert__(self) -> Constraint:
        from agentnet.constraints.composition import NotConstraint

        return NotConstraint(self)

    def check(self, candidate: Any, context: Any | None = None) -> bool:
        raise NotImplementedError

    def evaluate(self, candidate: Any, context: Any | None = None) -> ConstraintResult:
        passed = self.check(candidate, context)
        return ConstraintResult(
            constraint=self.name,
            passed=passed,
            kind=self.kind,
            message=None if passed else f"Constraint {self.name!r} failed",
        )

    def to_descriptor(self) -> ConstraintDescriptor:
        return ConstraintDescriptor(
            type=self.__class__.__name__,
            name=self.name,
            version=self.version,
            kind=self.kind,
            description=self.description,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_descriptor().to_dict()


def _coerce_constraint_kind(kind: ConstraintKind | str) -> ConstraintKind:
    try:
        return ConstraintKind(kind)
    except ValueError as exc:
        raise AgentNetConfigurationError(
            "Constraint kind must be 'hard' or 'soft'"
        ) from exc


def _coerce_constraint_version(version: str) -> str:
    version = str(version)
    if not version:
        raise AgentNetConfigurationError("Constraint version cannot be empty")
    return version
