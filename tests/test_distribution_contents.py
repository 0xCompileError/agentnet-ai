from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_distribution_uses_a_release_allowlist() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    included = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert included == {
        "/LICENSE",
        "/README.md",
        "/pyproject.toml",
        "/src/agentnet/**",
    }
