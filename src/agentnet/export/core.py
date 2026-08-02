"""Export AgentNet artifacts as installable Python packages."""

from __future__ import annotations

import keyword
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from agentnet._version import __version__
from agentnet.artifacts import validate_artifact
from agentnet.core import AgentNetConfigurationError, AgentNetValidationError

_PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:[A-Za-z0-9._+-]+)?$")


@dataclass(frozen=True, slots=True)
class PackageExportResult:
    """Result of exporting an artifact as a Python package."""

    path: Path
    package_name: str
    module_name: str
    artifact_path: Path
    pyproject_path: Path
    readme_path: Path
    loader_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_path": str(self.artifact_path),
            "loader_path": str(self.loader_path),
            "module_name": self.module_name,
            "package_name": self.package_name,
            "path": str(self.path),
            "pyproject_path": str(self.pyproject_path),
            "readme_path": str(self.readme_path),
        }


@dataclass(frozen=True, slots=True)
class PackageExporter:
    """Reusable package exporter configuration."""

    package_name: str
    module_name: str | None = None
    version: str = "0.1.0"
    description: str | None = None
    agentnet_requirement: str | None = None

    def export(
        self,
        artifact_path: str | Path,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> PackageExportResult:
        return export_package(
            artifact_path,
            output_dir,
            package_name=self.package_name,
            module_name=self.module_name,
            version=self.version,
            description=self.description,
            agentnet_requirement=self.agentnet_requirement,
            overwrite=overwrite,
        )


def export_package(
    artifact_path: str | Path,
    output_dir: str | Path,
    *,
    package_name: str,
    module_name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    agentnet_requirement: str | None = None,
    overwrite: bool = False,
) -> PackageExportResult:
    """Generate an installable Python package for an AgentNet artifact."""

    source_artifact = Path(artifact_path)
    package_root = Path(output_dir)
    normalized_package = _validate_package_name(package_name)
    normalized_module = _validate_module_name(
        module_name or _module_name_from_package(normalized_package)
    )
    _validate_version(version)
    _validate_artifact_ready(source_artifact)
    _prepare_output_dir(package_root, overwrite=overwrite)

    module_root = package_root / "src" / normalized_module
    artifact_target = module_root / "artifacts" / source_artifact.name
    module_root.mkdir(parents=True)
    artifact_target.parent.mkdir()
    shutil.copytree(source_artifact, artifact_target)

    pyproject_path = package_root / "pyproject.toml"
    readme_path = package_root / "README.md"
    init_path = module_root / "__init__.py"
    loader_path = module_root / "_loader.py"
    py_typed_path = module_root / "py.typed"

    effective_description = (
        description
        or f"Exported AgentNet package for {source_artifact.stem}."
    )
    effective_requirement = agentnet_requirement or f"agentnet>={__version__}"

    _write_text(
        pyproject_path,
        _render_pyproject(
            package_name=normalized_package,
            module_name=normalized_module,
            version=version,
            description=effective_description,
            agentnet_requirement=effective_requirement,
        ),
    )
    _write_text(
        readme_path,
        _render_readme(
            package_name=normalized_package,
            module_name=normalized_module,
            artifact_dir_name=source_artifact.name,
            description=effective_description,
        ),
    )
    _write_text(init_path, _render_init())
    _write_text(
        loader_path,
        _render_loader(
            package_name=normalized_package,
            module_name=normalized_module,
            artifact_dir_name=source_artifact.name,
        ),
    )
    _write_text(py_typed_path, "")

    return PackageExportResult(
        path=package_root,
        package_name=normalized_package,
        module_name=normalized_module,
        artifact_path=artifact_target,
        pyproject_path=pyproject_path,
        readme_path=readme_path,
        loader_path=loader_path,
    )


def _validate_artifact_ready(path: Path) -> None:
    result = validate_artifact(path)
    if result.passed:
        return
    messages = "; ".join(str(failure["message"]) for failure in result.failures)
    raise AgentNetValidationError(messages)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if path.is_file():
        if not overwrite:
            raise AgentNetConfigurationError(
                f"Output path {str(path)!r} already exists"
            )
        path.unlink()
        path.mkdir(parents=True)
        return
    if any(path.iterdir()):
        if not overwrite:
            raise AgentNetConfigurationError(
                f"Output path {str(path)!r} already exists and is not empty"
            )
        shutil.rmtree(path)
        path.mkdir(parents=True)


def _validate_package_name(package_name: str) -> str:
    normalized = package_name.strip()
    if not normalized or _PACKAGE_NAME_PATTERN.fullmatch(normalized) is None:
        raise AgentNetConfigurationError(f"Invalid package name {package_name!r}")
    return normalized


def _validate_module_name(module_name: str) -> str:
    normalized = module_name.strip()
    if (
        not normalized
        or _MODULE_NAME_PATTERN.fullmatch(normalized) is None
        or keyword.iskeyword(normalized)
    ):
        raise AgentNetConfigurationError(f"Invalid module name {module_name!r}")
    return normalized


def _validate_version(version: str) -> None:
    if _VERSION_PATTERN.fullmatch(version.strip()) is None:
        raise AgentNetConfigurationError(f"Invalid package version {version!r}")


def _module_name_from_package(package_name: str) -> str:
    return package_name.lower().replace("-", "_").replace(".", "_")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_pyproject(
    *,
    package_name: str,
    module_name: str,
    version: str,
    description: str,
    agentnet_requirement: str,
) -> str:
    return (
        "[build-system]\n"
        'requires = ["hatchling>=1.25"]\n'
        'build-backend = "hatchling.build"\n'
        "\n"
        "[project]\n"
        f"name = {_toml_string(package_name)}\n"
        f"version = {_toml_string(version)}\n"
        f"description = {_toml_string(description)}\n"
        'readme = "README.md"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = [\n'
        f"  {_toml_string(agentnet_requirement)},\n"
        "]\n"
        "\n"
        "[tool.hatch.build.targets.wheel]\n"
        f'packages = ["src/{module_name}"]\n'
        "\n"
        "[tool.hatch.build.targets.sdist]\n"
        "include = [\n"
        '  "README.md",\n'
        '  "pyproject.toml",\n'
        f'  "src/{module_name}",\n'
        "]\n"
    )


def _render_readme(
    *,
    package_name: str,
    module_name: str,
    artifact_dir_name: str,
    description: str,
) -> str:
    return (
        f"# {package_name}\n"
        "\n"
        f"{description}\n"
        "\n"
        f"This package embeds the AgentNet artifact `{artifact_dir_name}` and exposes a "
        "runtime loader. The package does not store live model clients, tool callables, "
        "MCP clients, or secrets.\n"
        "\n"
        "```python\n"
        "import agentnet as an\n"
        f"import {module_name}\n"
        "\n"
        "strong = an.FakeLLM(name=\"strong\", responses=[\"ok\"])\n"
        f"net = {module_name}.load(llms={{\"strong\": strong}})\n"
        "result = an.run(net, \"input\")\n"
        "```\n"
    )


def _render_init() -> str:
    return (
        '"""Generated AgentNet package loader."""\n'
        "\n"
        "from ._loader import (\n"
        "    ARTIFACT_DIR_NAME,\n"
        "    MODULE_NAME,\n"
        "    PACKAGE_NAME,\n"
        "    artifact_path,\n"
        "    load,\n"
        "    validate,\n"
        ")\n"
        "\n"
        "__all__ = [\n"
        '    "ARTIFACT_DIR_NAME",\n'
        '    "MODULE_NAME",\n'
        '    "PACKAGE_NAME",\n'
        '    "artifact_path",\n'
        '    "load",\n'
        '    "validate",\n'
        "]\n"
    )


def _render_loader(
    *,
    package_name: str,
    module_name: str,
    artifact_dir_name: str,
) -> str:
    return (
        '"""Runtime loader for an exported AgentNet artifact."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from collections.abc import Callable, Mapping\n"
        "from importlib.resources import as_file, files\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        "\n"
        "import agentnet as an\n"
        "\n"
        f"PACKAGE_NAME = {package_name!r}\n"
        f"MODULE_NAME = {module_name!r}\n"
        f"ARTIFACT_DIR_NAME = {artifact_dir_name!r}\n"
        "\n"
        "\n"
        "def _artifact_resource() -> Any:\n"
        '    return files(__package__) / "artifacts" / ARTIFACT_DIR_NAME\n'
        "\n"
        "\n"
        "def load(\n"
        "    *,\n"
        "    llms: Mapping[str, Any] | None = None,\n"
        "    tools: Mapping[str, Callable[..., Any]] | an.ToolRegistry | None = None,\n"
        "    mcp_servers: Mapping[str, Any] | an.MCPRegistry | None = None,\n"
        "    tracer: Any | None = None,\n"
        "    scheduler: Any | None = None,\n"
        ") -> an.Module:\n"
        '    """Load the embedded artifact with injected runtime dependencies."""\n'
        "\n"
        "    with as_file(_artifact_resource()) as artifact_dir:\n"
        "        return an.load(\n"
        "            artifact_dir,\n"
        "            llms=llms,\n"
        "            tools=tools,\n"
        "            mcp_servers=mcp_servers,\n"
        "            tracer=tracer,\n"
        "            scheduler=scheduler,\n"
        "        )\n"
        "\n"
        "\n"
        "def validate(\n"
        "    *,\n"
        "    llms: Mapping[str, Any] | None = None,\n"
        "    tools: Mapping[str, Callable[..., Any]] | an.ToolRegistry | None = None,\n"
        "    mcp_servers: Mapping[str, Any] | an.MCPRegistry | None = None,\n"
        ") -> an.ArtifactValidationResult:\n"
        '    """Validate the embedded artifact and optional dependency injection maps."""\n'
        "\n"
        "    with as_file(_artifact_resource()) as artifact_dir:\n"
        "        return an.validate_artifact(\n"
        "            artifact_dir,\n"
        "            llms=llms,\n"
        "            tools=tools,\n"
        "            mcp_servers=mcp_servers,\n"
        "        )\n"
        "\n"
        "\n"
        "def artifact_path() -> Path:\n"
        '    """Return the embedded artifact path for filesystem-based installs."""\n'
        "\n"
        "    resource = _artifact_resource()\n"
        "    if hasattr(resource, \"__fspath__\"):\n"
        "        return Path(resource)\n"
        "    with as_file(resource) as artifact_dir:\n"
        "        return Path(artifact_dir)\n"
    )
