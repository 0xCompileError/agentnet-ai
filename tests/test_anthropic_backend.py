from typing import Any

import pytest

import agentnet as an


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeAsyncClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:
        self.requests.append({"headers": headers, "json": json, "url": url})
        return FakeResponse(
            {
                "content": [{"text": "Hello from Anthropic", "type": "text"}],
                "model": "claude-3-5-sonnet-latest",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        )


@pytest.mark.anyio
async def test_anthropic_complete_posts_messages_request() -> None:
    client = FakeAsyncClient()
    llm = an.Anthropic(
        api_key="api-key",
        model="claude-3-5-sonnet-latest",
        client=client,
        max_tokens=512,
    )
    request = an.ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    response = await llm.complete(request)

    assert response == an.ChatResponse(
        content="Hello from Anthropic",
        finish_reason="end_turn",
        model="claude-3-5-sonnet-latest",
        usage={"input_tokens": 2, "output_tokens": 3},
    )
    assert client.requests == [
        {
            "headers": {
                "anthropic-version": "2023-06-01",
                "x-api-key": "api-key",
            },
            "json": {
                "max_tokens": 512,
                "messages": [{"content": "Hello", "role": "user"}],
                "model": "claude-3-5-sonnet-latest",
                "temperature": 0,
            },
            "url": "https://api.anthropic.com/v1/messages",
        }
    ]


def test_anthropic_satisfies_backend_protocol() -> None:
    llm = an.Anthropic(api_key="api-key", model="claude-3-5-sonnet-latest")

    assert isinstance(llm, an.LLMBackend)
    assert an.Anthropic.__name__ == "Anthropic"
