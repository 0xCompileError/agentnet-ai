"""Training budget tracking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from agentnet.core import AgentNetConfigurationError, AgentNetValidationError


@dataclass(slots=True, init=False)
class Budget:
    """Track coarse training limits for epochs, examples, trials, calls, and cost."""

    max_epochs: int | None
    max_examples: int | None
    max_trials: int | None
    max_cost: float | None
    max_llm_calls: int | None
    examples_used: int
    trials_used: int
    llm_calls_used: int
    cost_used: float

    def __init__(
        self,
        *,
        max_epochs: int | None = None,
        max_examples: int | None = None,
        max_trials: int | None = None,
        max_llm_calls: int | None = None,
        max_cost: float | None = None,
        examples_used: int = 0,
        trials_used: int = 0,
        llm_calls_used: int = 0,
        cost_used: float = 0.0,
    ) -> None:
        _validate_optional_positive_int(max_epochs, "max_epochs")
        _validate_optional_positive_int(max_examples, "max_examples")
        _validate_optional_positive_int(max_trials, "max_trials")
        _validate_optional_positive_int(max_llm_calls, "max_llm_calls")
        if max_cost is not None and max_cost < 0:
            raise AgentNetConfigurationError("Budget max_cost cannot be negative")
        if examples_used < 0:
            raise AgentNetConfigurationError("Budget examples_used cannot be negative")
        if trials_used < 0:
            raise AgentNetConfigurationError("Budget trials_used cannot be negative")
        if llm_calls_used < 0:
            raise AgentNetConfigurationError("Budget llm_calls_used cannot be negative")
        if cost_used < 0:
            raise AgentNetConfigurationError("Budget cost_used cannot be negative")

        self.max_epochs = max_epochs
        self.max_examples = max_examples
        self.max_trials = max_trials
        self.max_llm_calls = max_llm_calls
        self.max_cost = None if max_cost is None else float(max_cost)
        self.examples_used = int(examples_used)
        self.trials_used = int(trials_used)
        self.llm_calls_used = int(llm_calls_used)
        self.cost_used = float(cost_used)

    def can_run(
        self,
        *,
        epoch: int | None = None,
        examples: int = 0,
        trials: int = 0,
        llm_calls: int = 0,
        cost: float = 0.0,
    ) -> bool:
        if epoch is not None and epoch < 1:
            raise AgentNetConfigurationError("Budget epoch must be at least 1")
        if examples < 0:
            raise AgentNetConfigurationError("Budget examples cannot be negative")
        if trials < 0:
            raise AgentNetConfigurationError("Budget trials cannot be negative")
        if llm_calls < 0:
            raise AgentNetConfigurationError("Budget llm_calls cannot be negative")
        if cost < 0:
            raise AgentNetConfigurationError("Budget cost cannot be negative")

        if self.max_epochs is not None and epoch is not None and epoch > self.max_epochs:
            return False
        if self.max_examples is not None and self.examples_used + examples > self.max_examples:
            return False
        if self.max_trials is not None and self.trials_used + trials > self.max_trials:
            return False
        if (
            self.max_llm_calls is not None
            and self.llm_calls_used + llm_calls > self.max_llm_calls
        ):
            return False
        return not (
            self.max_cost is not None and self.cost_used + float(cost) > self.max_cost
        )

    def record(
        self,
        *,
        examples: int = 0,
        trials: int = 0,
        llm_calls: int = 0,
        cost: float = 0.0,
    ) -> None:
        if not self.can_run(
            examples=examples,
            trials=trials,
            llm_calls=llm_calls,
            cost=cost,
        ):
            raise AgentNetValidationError("Training budget would be exceeded")
        self.examples_used += int(examples)
        self.trials_used += int(trials)
        self.llm_calls_used += int(llm_calls)
        self.cost_used += float(cost)

    @property
    def remaining(self) -> dict[str, float | int | None]:
        return {
            "cost": None if self.max_cost is None else self.max_cost - self.cost_used,
            "epochs": self.max_epochs,
            "examples": (
                None if self.max_examples is None else self.max_examples - self.examples_used
            ),
            "llm_calls": (
                None
                if self.max_llm_calls is None
                else self.max_llm_calls - self.llm_calls_used
            ),
            "trials": None if self.max_trials is None else self.max_trials - self.trials_used,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_used": self.cost_used,
            "examples_used": self.examples_used,
            "llm_calls_used": self.llm_calls_used,
            "max_cost": self.max_cost,
            "max_epochs": self.max_epochs,
            "max_examples": self.max_examples,
            "max_llm_calls": self.max_llm_calls,
            "max_trials": self.max_trials,
            "trials_used": self.trials_used,
        }

    @classmethod
    def from_dict(cls, budget: Mapping[str, Any]) -> Self:
        return cls(
            max_epochs=_optional_int(budget.get("max_epochs")),
            max_examples=_optional_int(budget.get("max_examples")),
            max_trials=_optional_int(budget.get("max_trials")),
            max_llm_calls=_optional_int(budget.get("max_llm_calls")),
            max_cost=None if budget.get("max_cost") is None else float(budget["max_cost"]),
            examples_used=int(budget.get("examples_used", 0)),
            trials_used=int(budget.get("trials_used", 0)),
            llm_calls_used=int(budget.get("llm_calls_used", 0)),
            cost_used=float(budget.get("cost_used", 0.0)),
        )


BudgetManager = Budget


def _validate_optional_positive_int(value: int | None, field: str) -> None:
    if value is not None and value < 1:
        raise AgentNetConfigurationError(f"Budget {field} must be at least 1")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
