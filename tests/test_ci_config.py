from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_ci_runs_project_validation() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "name: CI" in workflow
    assert "uses: actions/checkout@v4" in workflow
    assert "uses: astral-sh/setup-uv@v5" in workflow
    assert "uses: actions/setup-python@v5" in workflow
    assert "python-version: '3.11'" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run pyright" in workflow
    assert "uv run pytest" in workflow
