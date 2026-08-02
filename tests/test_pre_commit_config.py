from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pre_commit_config_runs_project_validation_commands() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text()

    assert "repo: local" in config
    assert "id: ruff-check" in config
    assert "entry: uv run ruff check ." in config
    assert "id: pyright" in config
    assert "entry: uv run pyright" in config
    assert "id: pytest" in config
    assert "entry: uv run pytest" in config
