from typing import Any

import pytest

import agentnet as an


class FakeBedrockClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def converse(
        self,
        *,
        modelId: str,
        messages: list[dict[str, Any]],
        inferenceConfig: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "inferenceConfig": inferenceConfig,
                "messages": messages,
                "modelId": modelId,
            }
        )
        return {
            "output": {
                "message": {
                    "content": [{"text": "Hello from Bedrock"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 3},
        }


@pytest.mark.anyio
async def test_bedrock_complete_calls_converse() -> None:
    client = FakeBedrockClient()
    llm = an.Bedrock(
        client=client,
        max_tokens=512,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    request = an.ChatRequest(
        model="strong",
        messages=[{"content": "Hello", "role": "user"}],
        metadata={"temperature": 0},
    )

    response = await llm.complete(request)

    assert response == an.ChatResponse(
        content="Hello from Bedrock",
        finish_reason="end_turn",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        usage={"inputTokens": 2, "outputTokens": 3},
    )
    assert client.calls == [
        {
            "inferenceConfig": {"maxTokens": 512, "temperature": 0},
            "messages": [
                {
                    "content": [{"text": "Hello"}],
                    "role": "user",
                }
            ],
            "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        }
    ]


def test_bedrock_satisfies_backend_protocol() -> None:
    llm = an.Bedrock(
        client=FakeBedrockClient(),
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    assert isinstance(llm, an.LLMBackend)
    assert an.Bedrock.__name__ == "Bedrock"
