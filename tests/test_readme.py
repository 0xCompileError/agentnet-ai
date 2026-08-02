import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_has_readme_metadata_and_quickstart() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    readme = (ROOT / "README.md").read_text()

    assert pyproject["project"]["readme"] == "README.md"
    assert readme.startswith("# AgentNet")
    assert "## Why AgentNet" in readme
    assert "## What AgentNet Provides" in readme
    assert "## Quickstart" in readme
    assert "## Safety And Serialization Model" in readme
    assert "## Release Status" in readme
    assert "uv run pytest" in readme
    assert "import agentnet as an" in readme
    assert "an.train(" in readme
    assert "trained.run(" in readme
    assert "ReActAgent" in readme
    assert "Sequential" in readme
    assert ".agentnet" in readme
