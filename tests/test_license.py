import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_apache_2_license() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    license_text = (ROOT / "LICENSE").read_text()

    assert pyproject["project"]["license"] == "Apache-2.0"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "http://www.apache.org/licenses/" in license_text
