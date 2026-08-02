"""Built-in evaluation objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetValidationError,
    GraphResult,
)
from agentnet.core.schema import validate_schema
from agentnet.evaluation.base import EvaluationFailure, EvaluationResult, Objective


class SchemaObjective(Objective):
    """Objective that passes when candidate output satisfies a schema."""

    def __init__(
        self,
        schema: Any,
        *,
        name: str = "schema",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.schema = schema

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        del context
        output = _candidate_output(candidate)
        try:
            validate_schema(self.schema, output, label="candidate")
        except AgentNetValidationError as exc:
            return _failed_result(self.name, str(exc))
        return _passed_result(self.name)


class ExactMatchObjective(Objective):
    """Objective that requires exact equality with an expected value."""

    def __init__(
        self,
        expected: Any,
        *,
        field: str | None = None,
        name: str = "exact_match",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.expected = expected
        self.field = field

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        del context
        actual = (
            _lookup_field(candidate, self.field)
            if self.field is not None
            else _candidate_output(candidate)
        )
        if actual == self.expected:
            return _passed_result(self.name)
        return _failed_result(
            self.name,
            f"Expected {self.expected!r}, got {actual!r}",
        )


class ExpectedOutputObjective(Objective):
    """Compare a candidate with the current training example's expected output.

    This is the inferred objective used by :func:`agentnet.train`. String labels
    are deliberately forgiving of surrounding whitespace and casing while
    structured values retain normal Python equality semantics.
    """

    def __init__(
        self,
        *,
        normalize_strings: bool = True,
        name: str = "expected_output",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.normalize_strings = bool(normalize_strings)

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        if not isinstance(context, Mapping) or "expected_output" not in context:
            return _failed_result(
                self.name,
                "Expected-output evaluation requires training example context",
            )
        expected = context["expected_output"]
        actual = _candidate_output(candidate)
        if self.normalize_strings and isinstance(actual, str) and isinstance(expected, str):
            actual = actual.strip().casefold()
            expected = expected.strip().casefold()
        if actual == expected:
            return _passed_result(self.name)
        return _failed_result(
            self.name,
            "Candidate output did not match the expected output",
        )


class JudgeObjective(Objective):
    """Objective that delegates quality scoring to an explicitly injected judge."""

    def __init__(
        self,
        criteria: str,
        *,
        judge: Callable[[Any, str], Any] | None = None,
        threshold: float = 0.5,
        name: str = "judge",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if judge is not None and not callable(judge):
            raise AgentNetConfigurationError("JudgeObjective judge must be callable")
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.criteria = criteria
        self.judge = judge
        self.threshold = float(threshold)

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        del context
        if self.judge is None:
            return _failed_result(
                self.name,
                "JudgeObjective requires an injected judge callable",
            )

        try:
            raw_result = self.judge(_candidate_output(candidate), self.criteria)
        except Exception as exc:
            return _failed_result(
                self.name,
                f"Judge callable raised {type(exc).__name__}: {exc}",
            )

        result = _coerce_callable_result(
            raw_result,
            objective_name=self.name,
            threshold=self.threshold,
        )
        if result.passed:
            return result
        if result.failures:
            return result
        return EvaluationResult(
            score=result.score,
            passed=False,
            failures=[
                EvaluationFailure(
                    self.name,
                    f"Judge score {result.score} below threshold {self.threshold}",
                )
            ],
            metrics=result.metrics,
            metadata=result.metadata,
        )


class HumanFeedbackObjective(Objective):
    """Objective that scores numeric human feedback."""

    def __init__(
        self,
        *,
        min_score: float = 1.0,
        max_score: float = 5.0,
        field: str = "human_score",
        name: str = "human_feedback",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_score <= 0:
            raise AgentNetConfigurationError("HumanFeedbackObjective max_score must be positive")
        if min_score > max_score:
            raise AgentNetConfigurationError(
                "HumanFeedbackObjective min_score cannot exceed max_score"
            )
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.field = field

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        raw_score = _lookup_numeric(candidate, self.field, context)
        if raw_score is None:
            return _failed_result(self.name, f"Missing numeric feedback field {self.field!r}")

        normalized_score = raw_score / self.max_score
        passed = raw_score >= self.min_score
        return EvaluationResult(
            score=normalized_score,
            passed=passed,
            failures=()
            if passed
            else [
                EvaluationFailure(
                    self.name,
                    f"Human feedback score {raw_score} below minimum {self.min_score}",
                )
            ],
            metrics={
                f"{self.name}.normalized_score": normalized_score,
                f"{self.name}.score": raw_score,
            },
        )


class UnitTestObjective(Objective):
    """Objective that runs an explicitly supplied test callable."""

    def __init__(
        self,
        test: Callable[[Any], Any],
        *,
        name: str = "unit_test",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(test):
            raise AgentNetConfigurationError("UnitTestObjective test must be callable")
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.test = test

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        del context
        try:
            result = self.test(_candidate_output(candidate))
        except AssertionError as exc:
            message = str(exc) or "assert candidate failed unit test"
            return _failed_result(self.name, message)
        except Exception as exc:
            return _failed_result(
                self.name,
                f"Unit test raised {type(exc).__name__}: {exc}",
            )

        if result is False:
            return _failed_result(self.name, "Unit test returned False")
        return _passed_result(self.name)


class CostObjective(Objective):
    """Objective that rewards staying under a maximum cost."""

    def __init__(
        self,
        max_cost: float,
        *,
        field: str = "cost",
        name: str = "cost",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_cost < 0:
            raise AgentNetConfigurationError("CostObjective max_cost cannot be negative")
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.max_cost = float(max_cost)
        self.field = field

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        cost = _lookup_numeric(candidate, self.field, context)
        if cost is None:
            return _failed_result(self.name, f"Missing numeric cost field {self.field!r}")
        score = _remaining_budget_score(cost, self.max_cost)
        passed = cost <= self.max_cost
        return EvaluationResult(
            score=score,
            passed=passed,
            failures=()
            if passed
            else [
                EvaluationFailure(
                    self.name,
                    f"Cost {cost} exceeds maximum {self.max_cost}",
                )
            ],
            metrics={f"{self.name}.cost": cost, f"{self.name}.score": score},
        )


class LatencyObjective(Objective):
    """Objective that rewards staying under a latency target."""

    def __init__(
        self,
        p95_ms: float | None = None,
        *,
        max_latency_ms: float | None = None,
        field: str = "latency_ms",
        name: str = "latency",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        limit = max_latency_ms if max_latency_ms is not None else p95_ms
        if limit is None:
            raise AgentNetConfigurationError(
                "LatencyObjective requires p95_ms or max_latency_ms"
            )
        if limit < 0:
            raise AgentNetConfigurationError(
                "LatencyObjective latency limit cannot be negative"
            )
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.max_latency_ms = float(limit)
        self.field = field

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        latency_ms = _lookup_numeric(candidate, self.field, context)
        if latency_ms is None:
            return _failed_result(
                self.name,
                f"Missing numeric latency field {self.field!r}",
            )
        score = _remaining_budget_score(latency_ms, self.max_latency_ms)
        passed = latency_ms <= self.max_latency_ms
        return EvaluationResult(
            score=score,
            passed=passed,
            failures=()
            if passed
            else [
                EvaluationFailure(
                    self.name,
                    f"Latency {latency_ms} exceeds maximum {self.max_latency_ms}",
                )
            ],
            metrics={f"{self.name}.latency_ms": latency_ms, f"{self.name}.score": score},
        )


class ToolEfficiencyObjective(Objective):
    """Objective that scores tool completion under an optional call budget."""

    def __init__(
        self,
        *,
        max_tool_calls: int | None = None,
        min_completion_ratio: float = 1.0,
        name: str = "tool_efficiency",
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_tool_calls is not None and max_tool_calls < 0:
            raise AgentNetConfigurationError(
                "ToolEfficiencyObjective max_tool_calls cannot be negative"
            )
        if min_completion_ratio < 0 or min_completion_ratio > 1:
            raise AgentNetConfigurationError(
                "ToolEfficiencyObjective min_completion_ratio must be between 0 and 1"
            )
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.max_tool_calls = max_tool_calls
        self.min_completion_ratio = float(min_completion_ratio)

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        events = _lookup_tool_events(candidate, context)
        started = sum(1 for event in events if event.get("type") == "tool.called")
        completed = sum(1 for event in events if event.get("type") == "tool.completed")
        completion_ratio = 1.0 if started == 0 else completed / started
        if self.max_tool_calls is None:
            score = completion_ratio
        elif self.max_tool_calls == 0:
            score = 1.0 if completed == 0 else 0.0
        else:
            score = min(completed / self.max_tool_calls, 1.0)

        under_budget = self.max_tool_calls is None or started <= self.max_tool_calls
        passed = under_budget and completion_ratio >= self.min_completion_ratio
        failures: list[EvaluationFailure] = []
        if not under_budget:
            failures.append(
                EvaluationFailure(
                    self.name,
                    f"Tool call count {started} exceeds maximum {self.max_tool_calls}",
                )
            )
        if completion_ratio < self.min_completion_ratio:
            failures.append(
                EvaluationFailure(
                    self.name,
                    "Tool completion ratio "
                    f"{completion_ratio} below minimum {self.min_completion_ratio}",
                )
            )

        return EvaluationResult(
            score=score,
            passed=passed,
            failures=failures,
            metrics={
                f"{self.name}.completion_ratio": completion_ratio,
                f"{self.name}.completed": float(completed),
                f"{self.name}.started": float(started),
            },
        )


class CustomObjective(Objective):
    """Objective backed by an explicitly supplied in-process evaluator."""

    def __init__(
        self,
        name: str,
        evaluator: Callable[[Any, Any | None], Any],
        *,
        threshold: float = 0.0,
        weight: float = 1.0,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(evaluator):
            raise AgentNetConfigurationError("CustomObjective evaluator must be callable")
        super().__init__(
            name,
            description=description,
            metadata=metadata,
            weight=weight,
        )
        self.evaluator = evaluator
        self.threshold = float(threshold)

    def evaluate(self, candidate: Any, context: Any | None = None) -> EvaluationResult:
        try:
            raw_result = self.evaluator(_candidate_output(candidate), context)
        except Exception as exc:
            return _failed_result(
                self.name,
                f"Custom objective raised {type(exc).__name__}: {exc}",
            )
        return _coerce_callable_result(
            raw_result,
            objective_name=self.name,
            threshold=self.threshold,
        )


CustomMetricObjective = CustomObjective


def _passed_result(objective_name: str) -> EvaluationResult:
    return EvaluationResult(
        score=1.0,
        passed=True,
        metrics={f"{objective_name}.score": 1.0},
    )


def _failed_result(objective_name: str, message: str) -> EvaluationResult:
    return EvaluationResult(
        score=0.0,
        passed=False,
        failures=[EvaluationFailure(objective_name, message)],
        metrics={f"{objective_name}.score": 0.0},
    )


def _candidate_output(candidate: Any) -> Any:
    if isinstance(candidate, GraphResult):
        return candidate.output
    return candidate


def _lookup_numeric(candidate: Any, field: str, context: Any | None) -> float | None:
    value = _lookup_field(candidate, field, missing=None)
    if value is None and context is not None:
        value = _lookup_field(context, field, missing=None)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _lookup_field(candidate: Any, field: str | None, *, missing: Any = None) -> Any:
    if field is None:
        return _candidate_output(candidate)

    sources: list[Any] = [candidate]
    if isinstance(candidate, GraphResult):
        sources = [candidate.metadata, candidate.output, candidate]
    elif hasattr(candidate, "metadata"):
        sources = [candidate.metadata, candidate]

    for source in sources:
        value = _lookup_path(source, field, missing=missing)
        if value is not missing:
            return value
    return missing


def _lookup_path(value: Any, path: str, *, missing: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return missing
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return missing
    return current


def _remaining_budget_score(value: float, limit: float) -> float:
    if limit == 0:
        return 1.0 if value == 0 else 0.0
    return max(0.0, 1.0 - (value / limit))


def _lookup_tool_events(candidate: Any, context: Any | None) -> tuple[Mapping[str, Any], ...]:
    value = _lookup_field(candidate, "tool_events", missing=None)
    if value is None and context is not None:
        value = _lookup_field(context, "tool_events", missing=None)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return ()
    return tuple(event for event in value if isinstance(event, Mapping))


def _coerce_callable_result(
    raw_result: Any,
    *,
    objective_name: str,
    threshold: float,
) -> EvaluationResult:
    if isinstance(raw_result, EvaluationResult):
        return raw_result

    if isinstance(raw_result, Mapping):
        score = _coerce_score(raw_result.get("score", 0.0))
        passed = bool(raw_result.get("passed", score >= threshold))
        metadata = {
            str(key): value
            for key, value in raw_result.items()
            if key not in {"failures", "metrics", "passed", "score"}
        }
        metrics = _prefix_metrics(objective_name, raw_result.get("metrics", {}))
        metrics[f"{objective_name}.score"] = score
        return EvaluationResult(
            score=score,
            passed=passed,
            failures=()
            if passed
            else [
                EvaluationFailure(
                    objective_name,
                    f"Objective score {score} below threshold {threshold}",
                )
            ],
            metrics=metrics,
            metadata=metadata,
        )

    if isinstance(raw_result, bool):
        return _passed_result(objective_name) if raw_result else _failed_result(
            objective_name,
            "Objective returned False",
        )

    if isinstance(raw_result, int | float) and not isinstance(raw_result, bool):
        score = float(raw_result)
        passed = score >= threshold
        return EvaluationResult(
            score=score,
            passed=passed,
            failures=()
            if passed
            else [
                EvaluationFailure(
                    objective_name,
                    f"Objective score {score} below threshold {threshold}",
                )
            ],
            metrics={f"{objective_name}.score": score},
        )

    raise AgentNetConfigurationError(
        "Objective callable must return EvaluationResult, mapping, bool, or number"
    )


def _prefix_metrics(
    objective_name: str,
    metrics: Any,
) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    prefixed: dict[str, float] = {}
    for key, value in metrics.items():
        metric_name = str(key)
        if not metric_name.startswith(f"{objective_name}."):
            metric_name = f"{objective_name}.{metric_name}"
        prefixed[metric_name] = _coerce_score(value)
    return prefixed


def _coerce_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AgentNetConfigurationError("Objective scores and metrics must be numeric")
    return float(value)
