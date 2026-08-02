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
                        "message": {"content": "Hello from OpenAI"},
                    }
                ],
                "usage": {"completion_tokens": 3, "prompt_tokens": 2},
            }
        )


@pytest.mark.anyio
async def test_openai_complete_posts_chat_completions_request() -> None:
    client = FakeAsyncClient()
    llm = an.OpenAI(api_key="api-key", model="gpt-4o-mini", client=client)
    request = an.ChatRequest(
        model="cheap",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    response = await llm.complete(request)

    assert response == an.ChatResponse(
        content="Hello from OpenAI",
        finish_reason="stop",
        model="gpt-4o-mini",
        usage={"completion_tokens": 3, "prompt_tokens": 2},
    )
    assert client.requests == [
        {
            "headers": {"Authorization": "Bearer api-key"},
            "json": {
                "messages": [{"content": "Hello", "role": "user"}],
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
            "url": "https://api.openai.com/v1/chat/completions",
        }
    ]


def test_openai_satisfies_backend_protocol() -> None:
    llm = an.OpenAI(api_key="api-key", model="gpt-4o-mini")

    assert isinstance(llm, an.LLMBackend)
    assert an.OpenAI.__name__ == "OpenAI"
