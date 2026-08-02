# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


class FakeGatewayResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeGatewayClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeGatewayResponse:
        self.requests.append(
            {
                "model": json["model"],
                "url": url,
                "uses_authorization": "Authorization" in headers,
            }
        )
        return FakeGatewayResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Gateway response."},
                    }
                ],
                "usage": {"completion_tokens": 2, "prompt_tokens": 4},
            }
        )


def main() -> None:
    client = FakeGatewayClient()
    llm = an.LiteLLM(
        base_url="https://llm-gateway.internal",
        token="local-dev-token",
        model="gpt-4o-mini",
        name="gateway",
        client=client,
    )
    agent = an.ReActAgent(
        "gateway_agent",
        instructions="Use the configured gateway model.",
        llms=[llm],
    )

    output = an.run(agent, "Check gateway wiring.")
    emit(
        "litellm_gateway",
        {
            "model": client.requests[0]["model"],
            "output": output,
            "request_url": client.requests[0]["url"],
        },
    )


if __name__ == "__main__":
    main()
