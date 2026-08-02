import asyncio
from typing import Any

import pytest

import agentnet as an


class UpperModule(an.Module):
    async def arun(self, input: Any, context: Any | None = None) -> str:
        return str(input).upper()


class ContextModule(an.Module):
    async def arun(self, input: Any, context: Any | None = None) -> Any:
        return context


def test_package_run_executes_module_synchronously() -> None:
    module = UpperModule("upper")

    assert an.run(module, "hello") == "HELLO"


@pytest.mark.anyio
async def test_package_arun_executes_module_asynchronously() -> None:
    module = UpperModule("upper")

    assert await an.arun(module, "hello") == "HELLO"


def test_package_run_creates_context_when_missing() -> None:
    module = ContextModule("context")

    context = an.run(module, "hello")

    assert isinstance(context, an.RunContext)
    assert context.graph_state.run_id == context.run_id


def test_package_run_propagates_explicit_context() -> None:
    module = ContextModule("context")
    context = an.RunContext(run_id="run-1")

    assert an.run(module, "hello", context=context) is context


@pytest.mark.anyio
async def test_package_arun_creates_context_when_missing() -> None:
    module = ContextModule("context")

    context = await an.arun(module, "hello")

    assert isinstance(context, an.RunContext)
    assert context.graph_state.run_id == context.run_id


def test_package_run_rejects_cancelled_context() -> None:
    module = UpperModule("upper")
    context = an.RunContext(run_id="run-cancelled")
    context.cancel()

    with pytest.raises(asyncio.CancelledError):
        an.run(module, "hello", context=context)
