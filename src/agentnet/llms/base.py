"""Base LLM provider protocols."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from agentnet.llms.events import ChatEvent
from agentnet.llms.request import ChatRequest
from agentnet.llms.response import ChatResponse


@runtime_checkable
class LLMBackend(Protocol):
    """Provider-agnostic async chat backend protocol."""

    name: str
    model: str

    async def complete(self, request: ChatRequest) -> ChatResponse:
        ...

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        ...
