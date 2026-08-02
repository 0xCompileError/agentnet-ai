"""Vertex AI backend."""

from collections.abc import AsyncIterator
from inspect import isawaitable
from typing import Any

from agentnet.llms.events import ChatEvent
from agentnet.llms.request import ChatRequest
from agentnet.llms.response import ChatResponse


class VertexAI:
    """Google Vertex AI generate-content backend."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 1024,
        name: str = "vertexai",
    ) -> None:
        self._client = client
        self.max_tokens = max_tokens
        self.model = model
        self.name = name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        response = await _maybe_await(
            self._client.generate_content(
                model=self.model,
                contents=_to_vertex_contents(request.messages),
                generation_config={
                    "maxOutputTokens": self.max_tokens,
                    **request.metadata,
                },
            )
        )
        candidate = response["candidates"][0]
        return ChatResponse(
            content=_extract_text(candidate),
            finish_reason=candidate.get("finishReason"),
            model=self.model,
            usage=response.get("usageMetadata", {}),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent.from_response(await self.complete(request))


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


def _to_vertex_contents(messages: tuple[dict[str, str], ...]) -> list[dict[str, Any]]:
    return [
        {
            "parts": [{"text": message["content"]}],
            "role": message["role"],
        }
        for message in messages
    ]


def _extract_text(candidate: dict[str, Any]) -> str:
    parts = candidate.get("content", {}).get("parts", [])
    if not isinstance(parts, list):
        return ""

    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            text_parts.append(str(part["text"]))
    return "".join(text_parts)
