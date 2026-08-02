"""Lightweight schema validation primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

from agentnet.core.errors import AgentNetConfigurationError, AgentNetValidationError


@dataclass(frozen=True, slots=True, init=False)
class Schema:
    """Validate mapping values against Python type annotations."""

    fields: dict[str, Any]

    def __init__(self, fields: Mapping[str, Any]) -> None:
        object.__setattr__(self, "fields", dict(fields))

    def validate(self, value: Any, *, label: str = "value") -> Any:
        if not isinstance(value, Mapping):
            raise AgentNetValidationError(f"{label} must be a mapping")

        for field, expected in self.fields.items():
            field_label = f"{label}.{field}"
            if field not in value:
                raise AgentNetValidationError(f"{field_label} is required")
            _validate_type(value[field], expected, field_label)
        return value


def validate_schema(schema: Any | None, value: Any, *, label: str) -> Any:
    """Validate a value with a supported schema object or type annotation."""

    if schema is None:
        return value
    if isinstance(schema, Schema):
        return schema.validate(value, label=label)
    _validate_type(value, schema, label)
    return value


def _validate_type(value: Any, expected: Any, label: str) -> None:
    if expected is Any:
        return
    if expected is None:
        expected = type(None)
    if isinstance(expected, Schema):
        expected.validate(value, label=label)
        return
    if isinstance(expected, type):
        if not isinstance(value, expected):
            raise AgentNetValidationError(f"{label} must be {expected.__name__}")
        return

    origin = get_origin(expected)
    args = get_args(expected)
    if origin is list:
        _validate_list(value, args, label)
        return
    if origin is dict:
        _validate_dict(value, args, label)
        return
    if origin in (Union, UnionType):
        _validate_union(value, args, label)
        return

    raise AgentNetConfigurationError(f"Unsupported schema annotation for {label}: {expected!r}")


def _validate_list(value: Any, args: tuple[Any, ...], label: str) -> None:
    if not isinstance(value, list):
        raise AgentNetValidationError(f"{label} must be a list")
    if not args:
        return
    item_type = args[0]
    for index, item in enumerate(value):
        _validate_type(item, item_type, f"{label}[{index}]")


def _validate_dict(value: Any, args: tuple[Any, ...], label: str) -> None:
    if not isinstance(value, Mapping):
        raise AgentNetValidationError(f"{label} must be a mapping")
    if len(args) != 2:
        return
    key_type, value_type = args
    for key, item in value.items():
        _validate_type(key, key_type, f"{label}.key")
        _validate_type(item, value_type, f"{label}[{key!r}]")


def _validate_union(value: Any, args: tuple[Any, ...], label: str) -> None:
    errors: list[str] = []
    for option in args:
        try:
            _validate_type(value, option, label)
        except AgentNetValidationError as exc:
            errors.append(str(exc))
        else:
            return
    expected = " or ".join(_format_annotation(option) for option in args)
    raise AgentNetValidationError(f"{label} must be {expected}: {'; '.join(errors)}")


def _format_annotation(annotation: Any) -> str:
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    return repr(annotation)
