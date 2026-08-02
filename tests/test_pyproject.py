import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_defines_agentnet_project_metadata() -> None:
    pyproject_path = ROOT / "pyproject.toml"

    data = tomllib.loads(pyproject_path.read_text())

    assert data["project"]["name"] == "agentnet"
    assert data["project"]["version"] == "0.1.0"
    assert data["project"]["requires-python"] == ">=3.11"
    assert data["build-system"]["build-backend"] == "hatchling.build"
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["bypass-selection"] is True

    dev_dependencies = data["dependency-groups"]["dev"]
    assert any(dep.startswith("pytest") for dep in dev_dependencies)
    assert any(dep.startswith("ruff") for dep in dev_dependencies)
    assert any(dep.startswith("pyright") for dep in dev_dependencies)


def test_pyproject_defines_ruff_config() -> None:
    pyproject_path = ROOT / "pyproject.toml"

    data = tomllib.loads(pyproject_path.read_text())

    ruff_config = data["tool"]["ruff"]
    assert ruff_config["target-version"] == "py311"
    assert ruff_config["line-length"] == 100
    assert ruff_config["src"] == ["src", "tests"]
    assert ruff_config["lint"]["select"] == ["B", "E", "F", "I", "UP"]


def test_pyproject_configures_src_package_layout() -> None:
    pyproject_path = ROOT / "pyproject.toml"

    data = tomllib.loads(pyproject_path.read_text())

    wheel_target = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel_target["packages"] == ["src/agentnet"]


def test_pyproject_defines_pyright_config() -> None:
    pyproject_path = ROOT / "pyproject.toml"

    data = tomllib.loads(pyproject_path.read_text())

    pyright_config = data["tool"]["pyright"]
    assert pyright_config["include"] == ["src", "tests"]
    assert pyright_config["pythonVersion"] == "3.11"
    assert pyright_config["typeCheckingMode"] == "standard"


def test_pyproject_defines_pytest_config() -> None:
    pyproject_path = ROOT / "pyproject.toml"

    data = tomllib.loads(pyproject_path.read_text())

    pytest_config = data["tool"]["pytest"]["ini_options"]
    assert pytest_config["testpaths"] == ["tests"]
    assert pytest_config["python_files"] == ["test_*.py"]
    assert pytest_config["addopts"] == "--strict-config --strict-markers"


def test_pyproject_defines_coverage_config() -> None:
    pyproject_path = ROOT / "pyproject.toml"

    data = tomllib.loads(pyproject_path.read_text())

    dev_dependencies = data["dependency-groups"]["dev"]
    assert any(dep.startswith("pytest-cov") for dep in dev_dependencies)

    coverage_config = data["tool"]["coverage"]
    assert coverage_config["run"]["branch"] is True
    assert coverage_config["run"]["source"] == ["agentnet"]
    assert coverage_config["report"]["show_missing"] is True
    assert coverage_config["report"]["skip_covered"] is True
