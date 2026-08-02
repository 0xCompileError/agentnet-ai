from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_changelog_tracks_unreleased_foundation_work() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text()

    assert changelog.startswith("# Changelog")
    assert "## [Unreleased]" in changelog
    assert "### Added" in changelog
    assert "Foundation project scaffold" in changelog
