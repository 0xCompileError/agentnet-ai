from collections.abc import Callable
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


class FakeChatCompletionsClient:
    def __init__(self, content: str) -> None:
        self.content = content

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:
        return FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.content},
                    }
                ],
                "usage": {"completion_tokens": 1},
            }
        )


class FakeAnthropicClient:
    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:
        return FakeResponse(
            {
                "content": [{"text": "anthropic", "type": "text"}],
                "stop_reason": "end_turn",
                "usage": {"output_tokens": 1},
            }
        )


class FakeBedrockClient:
    async def converse(
        self,
        *,
        modelId: str,
        messages: list[dict[str, Any]],
        inferenceConfig: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "output": {"message": {"content": [{"text": "bedrock"}]}},
            "stopReason": "end_turn",
            "usage": {"outputTokens": 1},
        }


class FakeVertexAIClient:
    async def generate_content(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "vertexai"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"candidatesTokenCount": 1},
        }


def _request() -> an.ChatRequest:
    return an.ChatRequest(
        model="request-model",
        messages=[{"content": "Hello", "role": "user"}],
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda: an.FakeLLM(responses=["fake"]),
            an.ChatEvent(content="fake", delta="fake", model="request-model"),
        ),
        (
            lambda: an.LiteLLM(
                base_url="https://llm.example",
                client=FakeChatCompletionsClient("litellm"),
                model="litellm-model",
                token="token",
            ),
            an.ChatEvent(
                content="litellm",
                delta="litellm",
                finish_reason="stop",
                model="litellm-model",
                usage={"completion_tokens": 1},
            ),
        ),
        (
            lambda: an.OpenAI(
                api_key="api-key",
                client=FakeChatCompletionsClient("openai"),
                model="openai-model",
            ),
            an.ChatEvent(
                content="openai",
                delta="openai",
                finish_reason="stop",
                model="openai-model",
                usage={"completion_tokens": 1},
            ),
        ),
        (
            lambda: an.OpenAICompatible(
                api_key="api-key",
                base_url="https://compatible.example/v1",
                client=FakeChatCompletionsClient("compatible"),
                model="compatible-model",
            ),
            an.ChatEvent(
                content="compatible",
                delta="compatible",
                finish_reason="stop",
                model="compatible-model",
                usage={"completion_tokens": 1},
            ),
        ),
        (
            lambda: an.Anthropic(
                api_key="api-key",
                client=FakeAnthropicClient(),
                model="anthropic-model",
            ),
            an.ChatEvent(
                content="anthropic",
                delta="anthropic",
                finish_reason="end_turn",
                model="anthropic-model",
                usage={"output_tokens": 1},
            ),
        ),
        (
            lambda: an.Bedrock(client=FakeBedrockClient(), model="bedrock-model"),
            an.ChatEvent(
                content="bedrock",
                delta="bedrock",
                finish_reason="end_turn",
                model="bedrock-model",
                usage={"outputTokens": 1},
            ),
        ),
        (
            lambda: an.VertexAI(client=FakeVertexAIClient(), model="vertex-model"),
            an.ChatEvent(
                content="vertexai",
                delta="vertexai",
                finish_reason="STOP",
                model="vertex-model",
                usage={"candidatesTokenCount": 1},
            ),
        ),
    ],
)
async def test_builtin_providers_stream_chat_events(
    factory: Callable[[], Any],
    expected: an.ChatEvent,
) -> None:
    events = [event async for event in factory().stream(_request())]

    assert events == [expected]


def test_builtin_providers_are_exported_from_package_root() -> None:
    providers = {
        an.Anthropic,
        an.Bedrock,
        an.FakeLLM,
        an.LiteLLM,
        an.OpenAI,
        an.OpenAICompatible,
        an.VertexAI,
    }
    assert {provider.__name__ for provider in providers} == {
        "Anthropic",
        "Bedrock",
        "FakeLLM",
        "LiteLLM",
        "OpenAI",
        "OpenAICompatible",
        "VertexAI",
    }
