"""Amazon Bedrock backend."""

from collections.abc import AsyncIterator
from inspect import isawaitable
from typing import Any

from agentnet.llms.events import ChatEvent
from agentnet.llms.request import ChatRequest
from agentnet.llms.response import ChatResponse


class Bedrock:
    """Amazon Bedrock Runtime Converse backend."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 1024,
        name: str = "bedrock",
    ) -> None:
        self._client = client
        self.max_tokens = max_tokens
        self.model = model
        self.name = name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        result = await _maybe_await(
            self._client.converse(
                modelId=self.model,
                messages=_to_bedrock_messages(request.messages),
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    **request.metadata,
                },
            )
        )
        return ChatResponse(
            content=_extract_text(result),
            finish_reason=result.get("stopReason"),
            model=self.model,
            usage=result.get("usage", {}),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent.from_response(await self.complete(request))


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


def _to_bedrock_messages(messages: tuple[dict[str, str], ...]) -> list[dict[str, Any]]:
    return [
        {
            "content": [{"text": message["content"]}],
            "role": message["role"],
        }
        for message in messages
    ]


def _extract_text(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "".join(parts)
