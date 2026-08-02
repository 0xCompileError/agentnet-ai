"""Chat message builders for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentnet.agents.prompts import PromptRenderer
from agentnet.core import AgentState


@dataclass(frozen=True, slots=True)
class MessageBuilder:
    """Build provider-agnostic chat messages for an agent run."""

    instructions: str | None = None
    prompt_renderer: PromptRenderer = field(default_factory=PromptRenderer)

    def build(self, input: Any, *, state: AgentState | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.instructions:
            messages.append({"content": self.instructions, "role": "system"})
        if state is not None:
            messages.extend(
                {
                    "content": str(message["content"]),
                    "role": str(message["role"]),
                }
                for message in state.messages
            )
        messages.append(
            {"content": self.prompt_renderer.render(input, state=state), "role": "user"}
        )
        return messages
