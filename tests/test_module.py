from typing import Any

import pytest

import agentnet as an
from agentnet.core.module import Module


class EchoModule(Module):
    async def arun(self, input: Any, context: Any | None = None) -> dict[str, Any]:
        return {
            "context": context,
            "input": input,
            "name": self.name,
        }


def test_module_run_executes_async_implementation() -> None:
    module = EchoModule("echo")

    assert module.run("hello", context={"trace_id": "run-1"}) == {
        "context": {"trace_id": "run-1"},
        "input": "hello",
        "name": "echo",
    }


def test_module_state_dict_round_trips_name() -> None:
    module = EchoModule("original")

    state = module.state_dict()
    module.load_state_dict({"name": "restored"})

    assert state == {"name": "original"}
    assert module.name == "restored"


@pytest.mark.anyio
async def test_base_module_arun_requires_subclass_implementation() -> None:
    module = Module("base")

    with pytest.raises(NotImplementedError):
        await module.arun("input")


def test_module_is_exported_from_package_root() -> None:
    assert an.Module is Module
