"""Shared helpers for runnable AgentNet examples."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import agentnet as an


class CallableModule(an.Module):
    """Small module wrapper for examples that need deterministic transforms."""

    def __init__(self, name: str, handler: Callable[[Any], Any]) -> None:
        super().__init__(name)
        self.handler = handler

    async def arun(self, input: Any, context: object | None = None) -> Any:
        del context
        return self.handler(input)


class StaticModule(an.Module):
    """Module that returns a fixed value."""

    def __init__(self, name: str, output: Any) -> None:
        super().__init__(name)
        self.output = output

    async def arun(self, input: Any, context: object | None = None) -> Any:
        del input, context
        return self.output


def emit(example: str, result: Any, **extra: Any) -> None:
    """Print the stable JSON payload consumed by example tests."""

    payload = {
        "example": example,
        "ok": True,
        "result": result,
    }
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True))
