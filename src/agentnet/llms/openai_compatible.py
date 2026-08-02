"""Generic OpenAI-compatible backend."""

from typing import Any

from agentnet.llms.litellm import LiteLLM


class OpenAICompatible(LiteLLM):
    """Generic OpenAI chat-completions compatible backend."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        name: str = "openai-compatible",
        client: Any | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            token=api_key,
            model=model,
            name=name,
            client=client,
        )
