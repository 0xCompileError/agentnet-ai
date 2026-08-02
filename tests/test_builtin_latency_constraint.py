import agentnet as an
from agentnet.constraints import LatencyConstraint


class LatencyRecord:
    def __init__(self, latency_ms: float) -> None:
        self.latency_ms = latency_ms


def test_latency_constraint_accepts_numeric_latency_within_limit() -> None:
    constraint = LatencyConstraint(max_latency_ms=250)

    assert constraint.evaluate(200).passed is True
    assert constraint.evaluate(300).passed is False


def test_latency_constraint_reads_mapping_field() -> None:
    constraint = LatencyConstraint(max_latency_ms=250, field="p95_ms")

    assert constraint.evaluate({"p95_ms": 200}).passed is True
    assert constraint.evaluate({"p95_ms": 300}).passed is False


def test_latency_constraint_reads_object_attribute() -> None:
    constraint = LatencyConstraint(max_latency_ms=250)

    assert constraint.evaluate(LatencyRecord(200)).passed is True
    assert constraint.evaluate(LatencyRecord(300)).passed is False


def test_latency_constraint_is_exported_from_package_root() -> None:
    assert an.LatencyConstraint is LatencyConstraint
