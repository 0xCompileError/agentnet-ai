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
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Hello from a compatible API"},
                    }
                ],
                "usage": {"completion_tokens": 3, "prompt_tokens": 2},
            }
        )


@pytest.mark.anyio
async def test_openai_compatible_posts_chat_completions_request() -> None:
    client = FakeAsyncClient()
    llm = an.OpenAICompatible(
        api_key="api-key",
        base_url="https://compatible.example/v1/",
        client=client,
        model="model-a",
    )
    request = an.ChatRequest(
        model="compatible",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    response = await llm.complete(request)

    assert response == an.ChatResponse(
        content="Hello from a compatible API",
        finish_reason="stop",
        model="model-a",
        usage={"completion_tokens": 3, "prompt_tokens": 2},
    )
    assert client.requests == [
        {
            "headers": {"Authorization": "Bearer api-key"},
            "json": {
                "messages": [{"content": "Hello", "role": "user"}],
                "model": "model-a",
                "temperature": 0,
            },
            "url": "https://compatible.example/v1/chat/completions",
        }
    ]


def test_openai_compatible_satisfies_backend_protocol() -> None:
    llm = an.OpenAICompatible(
        api_key="api-key",
        base_url="https://compatible.example/v1",
        model="model-a",
    )

    assert isinstance(llm, an.LLMBackend)
    assert an.OpenAICompatible.__name__ == "OpenAICompatible"
