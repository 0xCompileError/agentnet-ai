"""OpenAI backend."""

from typing import Any

from agentnet.llms.litellm import LiteLLM


class OpenAI(LiteLLM):
    """Official OpenAI chat completions backend."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        name: str = "openai",
        client: Any | None = None,
    ) -> None:
        super().__init__(
            base_url="https://api.openai.com/v1",
            token=api_key,
            model=model,
            name=name,
            client=client,
        )
