from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_contributing_document_covers_local_validation() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text()

    assert contributing.startswith("# Contributing")
    assert "uv sync --dev" in contributing
    assert "uv run ruff check ." in contributing
    assert "uv run pyright" in contributing
    assert "uv run pytest" in contributing
