"""Prompt rendering for agent message construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentnet.core import AgentState


@dataclass(frozen=True, slots=True)
class PromptRenderer:
    """Render user input and prior agent state into prompt text."""

    def render(self, input: Any, *, state: AgentState | None = None) -> str:
        sections = [str(input)]
        if state is None:
            return sections[0]

        if state.reasoning:
            sections.append(_render_entries("Previous reasoning", state.reasoning, "content"))
        if state.actions:
            sections.append(_render_actions(state.actions))
        return "\n\n".join(sections)


def _render_entries(title: str, entries: list[dict[str, Any]], field: str) -> str:
    lines = [f"- {entry[field]}" for entry in entries]
    return f"{title}:\n" + "\n".join(lines)


def _render_actions(actions: list[dict[str, Any]]) -> str:
    lines = [f"- {_render_action(action)}" for action in actions]
    return "Previous actions:\n" + "\n".join(lines)


def _render_action(action: dict[str, Any]) -> str:
    name = str(action["name"])
    if "arguments" not in action:
        return name
    arguments = json.dumps(action["arguments"], sort_keys=True, default=str)
    return f"{name} {arguments}"
