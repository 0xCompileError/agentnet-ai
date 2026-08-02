import pytest

import agentnet as an
from agentnet.evaluation import EvaluationResult, ObjectiveSuite


def test_objective_framework_composes_and_aggregates_scores() -> None:
    objective = (
        an.SchemaObjective(an.Schema({"summary": str}), weight=2.0)
        + an.ExactMatchObjective({"summary": "ok"}, weight=1.0)
    )

    result = objective.evaluate({"summary": "ok"})
    failed = objective.evaluate({"summary": "different"})

    assert isinstance(objective, ObjectiveSuite)
    assert result == an.EvaluationResult(
        score=1.0,
        passed=True,
        failures=(),
        metrics={
            "exact_match.score": 1.0,
            "objective_count": 2.0,
            "schema.score": 1.0,
        },
        metadata={
            "evaluated_objectives": 2,
            "failed_objectives": 0,
        },
    )
    assert failed.score == pytest.approx(2.0 / 3.0)
    assert failed.passed is False
    assert failed.failures[0].objective == "exact_match"


def test_schema_objective_validates_candidate_output() -> None:
    objective = an.SchemaObjective(an.Schema({"summary": str, "risks": list[str]}))

    passed = objective.evaluate({"summary": "ok", "risks": ["latency"]})
    failed = objective.evaluate({"summary": "ok", "risks": ["latency", 3]})

    assert passed.score == 1.0
    assert passed.passed is True
    assert failed.score == 0.0
    assert failed.passed is False
    assert "candidate.risks[1]" in failed.failures[0].message


def test_exact_match_objective_supports_mapping_paths() -> None:
    objective = an.ExactMatchObjective(
        "approved",
        field="decision.status",
        name="status_match",
    )

    assert objective.evaluate({"decision": {"status": "approved"}}).passed is True
    failed = objective.evaluate({"decision": {"status": "rejected"}})
    assert failed.passed is False
    assert failed.metrics == {"status_match.score": 0.0}


def test_judge_objective_uses_injected_judge_callable() -> None:
    calls: list[tuple[object, str]] = []

    def judge(candidate: object, criteria: str) -> dict[str, object]:
        calls.append((candidate, criteria))
        return {"score": 0.8, "rationale": "clear and accurate"}

    objective = an.JudgeObjective(
        "Accurate and actionable.",
        judge=judge,
        threshold=0.75,
    )

    result = objective.evaluate("Ship the migration plan.")

    assert result.passed is True
    assert result.score == 0.8
    assert result.metrics == {"judge.score": 0.8}
    assert result.metadata == {"rationale": "clear and accurate"}
    assert calls == [("Ship the migration plan.", "Accurate and actionable.")]


def test_human_feedback_objective_reads_feedback_score() -> None:
    objective = an.HumanFeedbackObjective(min_score=4.0, max_score=5.0)

    result = objective.evaluate({"human_score": 4.5})
    failed = objective.evaluate({"human_score": 3.0})

    assert result.passed is True
    assert result.score == 0.9
    assert result.metrics == {
        "human_feedback.normalized_score": 0.9,
        "human_feedback.score": 4.5,
    }
    assert failed.passed is False
    assert failed.score == 0.6


def test_unit_test_objective_runs_explicit_test_callable() -> None:
    def assert_summary(candidate: object) -> None:
        assert isinstance(candidate, dict)
        assert candidate["summary"]

    objective = an.UnitTestObjective(assert_summary, name="summary_test")

    assert objective.evaluate({"summary": "ok"}).passed is True
    failed = objective.evaluate({"summary": ""})
    assert failed.passed is False
    assert "assert" in failed.failures[0].message


def test_cost_and_latency_objectives_read_result_metadata() -> None:
    result = an.GraphResult(
        output={"summary": "ok"},
        graph_state=an.GraphState("run-1"),
        metadata={"cost": 0.20, "latency_ms": 80.0},
    )

    cost = an.CostObjective(max_cost=0.25)
    latency = an.LatencyObjective(p95_ms=100)

    assert cost.evaluate(result).passed is True
    assert cost.evaluate(result).score == pytest.approx(0.2)
    assert latency.evaluate(result).passed is True
    assert latency.evaluate(result).score == pytest.approx(0.2)
    assert an.CostObjective(max_cost=0.10).evaluate(result).passed is False
    assert an.LatencyObjective(max_latency_ms=50).evaluate(result).passed is False


def test_tool_efficiency_objective_scores_tool_event_usage() -> None:
    context = an.RunContext("run-1")
    context.metadata["tool_events"] = [
        {"type": "tool.called", "tool": "search_docs"},
        {"type": "tool.completed", "tool": "search_docs"},
        {"type": "tool.called", "tool": "query_metrics"},
    ]
    objective = an.ToolEfficiencyObjective(
        max_tool_calls=3,
        min_completion_ratio=0.5,
    )

    result = objective.evaluate("output", context=context)

    assert result.passed is True
    assert result.score == pytest.approx((2.0 / 3.0) * (1.0 / 2.0))
    assert result.metrics == {
        "tool_efficiency.completion_ratio": 0.5,
        "tool_efficiency.completed": 1.0,
        "tool_efficiency.started": 2.0,
    }


def test_custom_metric_objective_wraps_explicit_callable() -> None:
    objective = an.CustomMetricObjective(
        "brevity",
        evaluator=lambda candidate, context: {
            "score": 0.75,
            "passed": len(str(candidate)) < 20,
            "metrics": {"tokens": float(len(str(candidate).split()))},
        },
    )

    result = objective.evaluate("short answer")

    assert result == an.EvaluationResult(
        score=0.75,
        passed=True,
        failures=(),
        metrics={"brevity.tokens": 2.0, "brevity.score": 0.75},
    )


def test_score_aggregation_combines_results_with_weights() -> None:
    result = an.aggregate_evaluation_results(
        [
            an.EvaluationResult(score=1.0, passed=True, metrics={"a.score": 1.0}),
            an.EvaluationResult(
                score=0.0,
                passed=False,
                failures=[an.EvaluationFailure("b", "failed")],
                metrics={"b.score": 0.0},
            ),
        ],
        weights=[3.0, 1.0],
    )

    assert result.score == 0.75
    assert result.passed is False
    assert result.failures == (an.EvaluationFailure("b", "failed"),)
    assert result.metrics == {
        "a.score": 1.0,
        "b.score": 0.0,
        "objective_count": 2.0,
    }
    assert result.metadata == {
        "evaluated_objectives": 2,
        "failed_objectives": 1,
    }


def test_evaluation_result_round_trips_to_dict() -> None:
    result = an.EvaluationResult(
        score=0.5,
        passed=False,
        failures=[an.EvaluationFailure("judge", "too vague")],
        metrics={"judge.score": 0.5},
        metadata={"rationale": "too vague"},
    )

    restored = EvaluationResult.from_dict(result.to_dict())

    assert restored == result


def test_evaluation_objectives_are_exported_from_package_root() -> None:
    assert an.EvaluationResult is EvaluationResult
    assert an.ObjectiveSuite is ObjectiveSuite
    assert an.Objective is not None
    assert an.SchemaObjective is not None
    assert an.JudgeObjective is not None
    assert an.ExactMatchObjective is not None
    assert an.HumanFeedbackObjective is not None
    assert an.UnitTestObjective is not None
    assert an.CostObjective is not None
    assert an.LatencyObjective is not None
    assert an.ToolEfficiencyObjective is not None
    assert an.CustomObjective is not None
    assert an.CustomMetricObjective is an.CustomObjective
