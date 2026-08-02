"""Training datasets."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, overload

from agentnet.core import AgentNetConfigurationError
from agentnet.mcp._security import validate_safe_metadata


@dataclass(frozen=True, slots=True, init=False)
class TrainingExample:
    """One training or evaluation case."""

    input: Any
    expected_output: Any | None
    id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        input: Any,
        *,
        expected_output: Any | None = None,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if id is not None and not id:
            raise AgentNetConfigurationError("TrainingExample id cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="TrainingExample")

        object.__setattr__(self, "input", input)
        object.__setattr__(self, "expected_output", expected_output)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "metadata", metadata_copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_output": self.expected_output,
            "id": self.id,
            "input": self.input,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, example: Mapping[str, Any]) -> Self:
        return cls(
            example.get("input"),
            expected_output=example.get("expected_output"),
            id=None if example.get("id") is None else str(example["id"]),
            metadata=dict(example.get("metadata", {})),
        )


class Dataset(Sequence[TrainingExample]):
    """Immutable collection of training examples."""

    def __init__(
        self,
        examples: Iterable[TrainingExample | Mapping[str, Any]],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if name is not None and not name:
            raise AgentNetConfigurationError("Dataset name cannot be empty")
        metadata_copy = dict(metadata or {})
        validate_safe_metadata(metadata_copy, label="Dataset")
        self.name = name
        self.metadata = metadata_copy
        self.examples = tuple(_coerce_example(example) for example in examples)

    @overload
    def __getitem__(self, index: int) -> TrainingExample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrainingExample, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> TrainingExample | tuple[TrainingExample, ...]:
        return self.examples[index]

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": [example.to_dict() for example in self.examples],
            "metadata": self.metadata.copy(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, dataset: Mapping[str, Any]) -> Self:
        return cls(
            [
                TrainingExample.from_dict(dict(example))
                for example in dataset.get("examples", ())
            ],
            name=None if dataset.get("name") is None else str(dataset["name"]),
            metadata=dict(dataset.get("metadata", {})),
        )


def _coerce_example(example: TrainingExample | Mapping[str, Any]) -> TrainingExample:
    if isinstance(example, TrainingExample):
        return example
    return TrainingExample.from_dict(example)
