"""Deterministic fake LLM backend for tests and examples."""

from collections.abc import AsyncIterator, Sequence

from agentnet.llms.events import ChatEvent
from agentnet.llms.request import ChatRequest
from agentnet.llms.response import ChatResponse


class FakeLLM:
    """LLM backend that returns queued responses without network calls."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        name: str = "fake",
        model: str = "fake-model",
    ) -> None:
        self.name = name
        self.model = model
        self.requests: list[ChatRequest] = []
        self._responses = list(responses or [""])

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        content = self._responses.pop(0) if self._responses else ""
        return ChatResponse(content=content, model=request.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent.from_response(await self.complete(request))
