"""Shared MCP descriptor safety checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentnet.core import AgentNetConfigurationError

_SECRET_KEY_PARTS = ("api_key", "password", "secret", "token")


def validate_safe_metadata(metadata: Mapping[str, Any], *, label: str) -> None:
    for key, value in metadata.items():
        _validate_safe_key(key, label=label)
        _validate_nested_metadata(value, label=label)


def validate_descriptor_payload_no_secrets(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() == "env":
                raise AgentNetConfigurationError(
                    f"{label} cannot include env because it may serialize secrets"
                )
            _validate_safe_key(key, label=label)
            validate_descriptor_payload_no_secrets(nested, label=label)
        return

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for nested in value:
            validate_descriptor_payload_no_secrets(nested, label=label)


def _validate_nested_metadata(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        validate_safe_metadata(value, label=label)
        return

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for nested in value:
            _validate_nested_metadata(nested, label=label)


def _validate_safe_key(key: object, *, label: str) -> None:
    normalized = str(key).lower()
    if any(part in normalized for part in _SECRET_KEY_PARTS):
        raise AgentNetConfigurationError(
            f"{label} metadata key {key!r} may serialize secrets"
        )
