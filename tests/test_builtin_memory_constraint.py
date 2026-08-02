import agentnet as an
from agentnet.constraints import MemoryConstraint


class MemoryRecord:
    def __init__(self, memory_mb: float) -> None:
        self.memory_mb = memory_mb


def test_memory_constraint_accepts_numeric_memory_within_limit() -> None:
    constraint = MemoryConstraint(max_memory_mb=512)

    assert constraint.evaluate(256).passed is True
    assert constraint.evaluate(1024).passed is False


def test_memory_constraint_reads_mapping_field() -> None:
    constraint = MemoryConstraint(max_memory_mb=512, field="peak_memory_mb")

    assert constraint.evaluate({"peak_memory_mb": 256}).passed is True
    assert constraint.evaluate({"peak_memory_mb": 1024}).passed is False


def test_memory_constraint_reads_object_attribute() -> None:
    constraint = MemoryConstraint(max_memory_mb=512)

    assert constraint.evaluate(MemoryRecord(256)).passed is True
    assert constraint.evaluate(MemoryRecord(1024)).passed is False


def test_memory_constraint_is_exported_from_package_root() -> None:
    assert an.MemoryConstraint is MemoryConstraint
