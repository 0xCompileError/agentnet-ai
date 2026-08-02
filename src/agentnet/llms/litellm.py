"""LiteLLM-compatible backend."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentnet.llms.events import ChatEvent
from agentnet.llms.request import ChatRequest
from agentnet.llms.response import ChatResponse


class LiteLLM:
    """OpenAI-compatible LiteLLM gateway backend."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        model: str,
        name: str = "litellm",
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.name = name
        self._client = client

    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "messages": [message.copy() for message in request.messages],
            "model": self.model,
            **request.metadata,
        }
        response = await self._post(payload)
        data = response.json()
        choice = data["choices"][0]
        return ChatResponse(
            content=str(choice["message"]["content"]),
            finish_reason=choice.get("finish_reason"),
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
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )
            response.raise_for_status()
            return response
        finally:
            if should_close:
                await client.aclose()
