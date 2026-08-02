from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_codeowners_protects_repository_and_workflows() -> None:
    entries = {
        line.strip()
        for line in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "* @0xCompileError" in entries
    assert "/.github/ @0xCompileError" in entries
    assert "/.github/workflows/ @0xCompileError" in entries


def test_validate_job_runs_for_pull_requests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "\n  validate:\n" in workflow
