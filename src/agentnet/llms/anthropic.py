"""Anthropic backend."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentnet.llms.events import ChatEvent
from agentnet.llms.request import ChatRequest
from agentnet.llms.response import ChatResponse


class Anthropic:
    """Anthropic Messages API backend."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int = 1024,
        name: str = "anthropic",
        client: Any | None = None,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self.api_key = api_key
        self.anthropic_version = anthropic_version
        self.max_tokens = max_tokens
        self.model = model
        self.name = name
        self._client = client

    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "max_tokens": self.max_tokens,
            "messages": [message.copy() for message in request.messages],
            "model": self.model,
            **request.metadata,
        }
        response = await self._post(payload)
        data = response.json()
        return ChatResponse(
            content=_extract_text(data.get("content", [])),
            finish_reason=data.get("stop_reason"),
            model=self.model,
            usage=data.get("usage", {}),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent.from_response(await self.complete(request))

    async def _post(self, payload: dict[str, Any]) -> Any:
        client = self._client
        should_close = client is None
        if client is None:
            client = httpx.AsyncClient()

        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "anthropic-version": self.anthropic_version,
                    "x-api-key": self.api_key,
                },
                json=payload,
            )
            response.raise_for_status()
            return response
        finally:
            if should_close:
                await client.aclose()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
