from typing import Any

import pytest

import agentnet as an
from agentnet.llms.litellm import LiteLLM


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
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Hello from LiteLLM"},
                    }
                ],
                "usage": {"completion_tokens": 3, "prompt_tokens": 2},
            }
        )


@pytest.mark.anyio
async def test_litellm_complete_posts_openai_compatible_request() -> None:
    client = FakeAsyncClient()
    llm = LiteLLM(
        base_url="https://llm-gateway.example",
        token="token",
        model="gpt-4o",
        client=client,
    )
    request = an.ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    response = await llm.complete(request)

    assert response == an.ChatResponse(
        content="Hello from LiteLLM",
        finish_reason="stop",
        model="gpt-4o",
        usage={"completion_tokens": 3, "prompt_tokens": 2},
    )
    assert client.requests == [
        {
            "headers": {"Authorization": "Bearer token"},
            "json": {
                "messages": [{"content": "Hello", "role": "user"}],
                "model": "gpt-4o",
                "temperature": 0,
            },
            "url": "https://llm-gateway.example/chat/completions",
        }
    ]


def test_litellm_satisfies_backend_protocol() -> None:
    llm = LiteLLM(base_url="https://example.com", token="token", model="model")

    assert isinstance(llm, an.LLMBackend)
    assert an.LiteLLM is LiteLLM
