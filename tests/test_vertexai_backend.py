from typing import Any

import pytest

import agentnet as an


class FakeVertexAIClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "contents": contents,
                "generation_config": generation_config,
                "model": model,
            }
        )
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from VertexAI"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "candidatesTokenCount": 3,
                "promptTokenCount": 2,
            },
        }


@pytest.mark.anyio
async def test_vertexai_complete_calls_generate_content() -> None:
    client = FakeVertexAIClient()
    llm = an.VertexAI(client=client, max_tokens=512, model="gemini-1.5-pro")
    request = an.ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    response = await llm.complete(request)

    assert response == an.ChatResponse(
        content="Hello from VertexAI",
        finish_reason="STOP",
        model="gemini-1.5-pro",
        usage={"candidatesTokenCount": 3, "promptTokenCount": 2},
    )
    assert client.calls == [
        {
            "contents": [
                {
                    "parts": [{"text": "Hello"}],
                    "role": "user",
                }
            ],
            "generation_config": {"maxOutputTokens": 512, "temperature": 0},
            "model": "gemini-1.5-pro",
        }
    ]


def test_vertexai_satisfies_backend_protocol() -> None:
    llm = an.VertexAI(client=FakeVertexAIClient(), model="gemini-1.5-pro")

    assert isinstance(llm, an.LLMBackend)
    assert an.VertexAI.__name__ == "VertexAI"
