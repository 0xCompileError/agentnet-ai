"""Runtime state containers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self


@dataclass(slots=True)
class AgentState:
    """Mutable execution state for a single agent."""

    name: str
    step: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"content": content, "role": role})

    def add_reasoning(
        self,
        content: str,
        *,
        model: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        reasoning: dict[str, Any] = {"content": content, "step": self.step}
        if model is not None:
            reasoning["model"] = model
        if metadata:
            reasoning["metadata"] = dict(metadata)
        self.reasoning.append(reasoning)

    def add_action(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        action: dict[str, Any] = {"name": name, "step": self.step}
        if arguments is not None:
            action["arguments"] = dict(arguments)
        if metadata:
            action["metadata"] = dict(metadata)
        self.actions.append(action)

    def advance(self) -> None:
        self.step += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.copy() for action in self.actions],
            "messages": [message.copy() for message in self.messages],
            "metadata": self.metadata.copy(),
            "name": self.name,
            "reasoning": [reasoning.copy() for reasoning in self.reasoning],
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Self:
        return cls(
            name=str(state["name"]),
            step=int(state.get("step", 0)),
            actions=[dict(action) for action in state.get("actions", [])],
            messages=[dict(message) for message in state.get("messages", [])],
            reasoning=[dict(reasoning) for reasoning in state.get("reasoning", [])],
            metadata=dict(state.get("metadata", {})),
        )


@dataclass(slots=True)
class GraphState:
    """Mutable execution state for an agent graph run."""

    run_id: str
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_agent_state(self, state: AgentState) -> None:
        self.agent_states[state.name] = state

    def get_agent_state(self, name: str) -> AgentState:
        return self.agent_states[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_states": {
                name: agent_state.to_dict()
                for name, agent_state in self.agent_states.items()
            },
            "metadata": self.metadata.copy(),
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Self:
        return cls(
            run_id=str(state["run_id"]),
            agent_states={
                name: AgentState.from_dict(agent_state)
                for name, agent_state in state.get("agent_states", {}).items()
            },
            metadata=dict(state.get("metadata", {})),
        )
