from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_scaffold_has_index_and_navigation() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text()
    index = (ROOT / "docs" / "index.md").read_text()

    assert "site_name: AgentNet" in mkdocs
    assert "nav:" in mkdocs
    assert "- Home: index.md" in mkdocs
    assert index.startswith("# AgentNet Documentation")
    assert "## Project Status" in index
