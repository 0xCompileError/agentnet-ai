from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_security_policy_defines_reporting_process() -> None:
    security = (ROOT / "SECURITY.md").read_text()

    assert security.startswith("# Security Policy")
    assert "## Supported Versions" in security
    assert "## Reporting a Vulnerability" in security
    assert "Do not report security vulnerabilities in public issues" in security
