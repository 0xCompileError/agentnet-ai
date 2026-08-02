import agentnet as an
from agentnet.constraints import CostConstraint


class CostRecord:
    def __init__(self, cost: float) -> None:
        self.cost = cost


def test_cost_constraint_accepts_numeric_costs_within_limit() -> None:
    constraint = CostConstraint(max_cost=0.25)

    assert constraint.evaluate(0.10).passed is True
    assert constraint.evaluate(0.30).passed is False


def test_cost_constraint_reads_mapping_field() -> None:
    constraint = CostConstraint(max_cost=1.0, field="total_cost")

    assert constraint.evaluate({"total_cost": 0.5}).passed is True
    assert constraint.evaluate({"total_cost": 1.5}).passed is False


def test_cost_constraint_reads_object_attribute() -> None:
    constraint = CostConstraint(max_cost=0.25)

    assert constraint.evaluate(CostRecord(0.10)).passed is True
    assert constraint.evaluate(CostRecord(0.30)).passed is False


def test_cost_constraint_is_exported_from_package_root() -> None:
    assert an.CostConstraint is CostConstraint
