from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

import agentnet as an

ROOT = Path(__file__).resolve().parents[1]


def _make_artifact(tmp_path: Path, name: str = "decision_net") -> Path:
    artifact_path = tmp_path / f"{name}.agentnet"
    an.save(
        an.ReActAgent(
            "planner",
            instructions="Plan clearly.",
            llms=["strong"],
        ),
        artifact_path,
        name=name,
        metadata={"owner": "platform"},
    )
    return artifact_path


def _import_from_src(module_name: str, source_root: Path) -> Any:
    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(source_root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(source_root))


def test_export_package_generates_python_project_templates(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path)
    package_path = tmp_path / "exported"

    result = an.export_package(
        artifact_path,
        package_path,
        package_name="decision-net",
        description="Decision support network",
    )

    assert result.path == package_path
    assert result.package_name == "decision-net"
    assert result.module_name == "decision_net"
    assert result.artifact_path == (
        package_path
        / "src"
        / "decision_net"
        / "artifacts"
        / "decision_net.agentnet"
    )
    assert (package_path / "README.md").is_file()
    assert (package_path / "pyproject.toml").is_file()
    assert (package_path / "src" / "decision_net" / "__init__.py").is_file()
    assert (package_path / "src" / "decision_net" / "_loader.py").is_file()
    assert (result.artifact_path / "manifest.json").is_file()

    pyproject = tomllib.loads((package_path / "pyproject.toml").read_text())
    assert pyproject["project"]["name"] == "decision-net"
    assert pyproject["project"]["description"] == "Decision support network"
    assert pyproject["project"]["dependencies"] == [f"agentnet>={an.__version__}"]
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/decision_net"
    ]

    readme = (package_path / "README.md").read_text()
    assert "decision-net" in readme
    assert "decision_net.load" in readme
    assert "llms" in readme


def test_generated_package_loader_validates_and_loads_artifact(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path, name="support_net")
    package_path = tmp_path / "support_package"
    an.export_package(
        artifact_path,
        package_path,
        package_name="support-net",
    )

    support_net = _import_from_src("support_net", package_path / "src")
    llm = an.FakeLLM(responses=["ok"], name="strong")

    validation = support_net.validate(llms={"strong": llm})
    loaded = support_net.load(llms={"strong": llm})

    assert validation.passed is True
    assert loaded.instructions == "Plan clearly."
    assert an.run(loaded, "input") == "ok"


def test_export_package_rejects_invalid_names_and_existing_outputs(
    tmp_path: Path,
) -> None:
    artifact_path = _make_artifact(tmp_path)
    package_path = tmp_path / "exported"
    package_path.mkdir()
    (package_path / "existing.txt").write_text("keep")

    with pytest.raises(an.AgentNetConfigurationError, match="already exists"):
        an.export_package(
            artifact_path,
            package_path,
            package_name="decision-net",
            overwrite=False,
        )

    with pytest.raises(an.AgentNetConfigurationError, match="Invalid package name"):
        an.export_package(
            artifact_path,
            tmp_path / "bad_dist",
            package_name="bad name",
        )

    with pytest.raises(an.AgentNetConfigurationError, match="Invalid module name"):
        an.export_package(
            artifact_path,
            tmp_path / "bad_module",
            package_name="decision-net",
            module_name="class",
        )


def test_exported_package_executes_from_source_layout(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path, name="install_net")
    package_path = tmp_path / "install_package"
    an.export_package(
        artifact_path,
        package_path,
        package_name="install-net",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(package_path / "src"), str(ROOT / "src")]
    )
    execution = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import agentnet as an, install_net; "
                "net = install_net.load("
                "llms={'strong': an.FakeLLM(responses=['ok'], "
                "name='strong')}"
                "); "
                "print(an.run(net, 'input'))"
            ),
        ],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )

    assert execution.stdout.strip() == "ok"


def test_package_export_public_exports_are_available() -> None:
    exported: list[Any] = [
        an.PackageExportResult,
        an.PackageExporter,
        an.export_package,
    ]

    assert all(value is not None for value in exported)
