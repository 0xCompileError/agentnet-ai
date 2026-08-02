from collections.abc import AsyncIterator
from typing import Any

import agentnet as an
from agentnet.llms.base import LLMBackend


class Provider:
    name = "fake"
    model = "fake-model"

    async def complete(self, request: Any) -> dict[str, Any]:
        return {"content": request}

    async def stream(self, request: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"content": request}


def test_llm_backend_protocol_is_runtime_checkable() -> None:
    assert isinstance(Provider(), LLMBackend)


def test_llm_backend_protocol_is_exported_from_package_root() -> None:
    assert an.LLMBackend is LLMBackend
