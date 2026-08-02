import agentnet as an
from agentnet.constraints import TokenConstraint


class TokenRecord:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens


def test_token_constraint_accepts_numeric_token_counts_within_limit() -> None:
    constraint = TokenConstraint(max_tokens=100)

    assert constraint.evaluate(50).passed is True
    assert constraint.evaluate(101).passed is False


def test_token_constraint_reads_mapping_field() -> None:
    constraint = TokenConstraint(max_tokens=100, field="total_tokens")

    assert constraint.evaluate({"total_tokens": 80}).passed is True
    assert constraint.evaluate({"total_tokens": 120}).passed is False


def test_token_constraint_reads_object_attribute() -> None:
    constraint = TokenConstraint(max_tokens=100)

    assert constraint.evaluate(TokenRecord(80)).passed is True
    assert constraint.evaluate(TokenRecord(120)).passed is False


def test_token_constraint_is_exported_from_package_root() -> None:
    assert an.TokenConstraint is TokenConstraint
