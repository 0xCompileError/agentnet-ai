from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_code_of_conduct_defines_standards_and_enforcement() -> None:
    code_of_conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text()

    assert code_of_conduct.startswith("# Code of Conduct")
    assert "## Our Standards" in code_of_conduct
    assert "## Unacceptable Behavior" in code_of_conduct
    assert "## Enforcement" in code_of_conduct
