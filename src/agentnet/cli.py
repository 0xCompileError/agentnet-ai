"""Command-line interface for descriptor-safe AgentNet workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentnet._version import __version__
from agentnet.artifacts import validate_artifact
from agentnet.core import AgentNetError
from agentnet.evaluation import EvaluationFailure, EvaluationResult
from agentnet.export import export_package
from agentnet.mcp._security import validate_descriptor_payload_no_secrets
from agentnet.training import Dataset, TrainingHistory, TrainingStep

_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AgentNet CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (AgentNetError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"agentnet: error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the AgentNet CLI parser."""

    parser = argparse.ArgumentParser(
        prog="agentnet",
        description=(
            "AgentNet command-line tools. Common commands: agentnet init, "
            "agentnet doctor, agentnet inspect, agentnet export, agentnet eval, "
            "agentnet train."
        ),
    )
    parser.add_argument("--version", action="version", version=f"agentnet {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    init_parser = subcommands.add_parser("init", help="Create a new AgentNet project scaffold.")
    init_parser.add_argument("path", nargs="?", default=".", help="Project directory to create.")
    init_parser.add_argument("--name", help="Project name. Defaults to the directory name.")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite AgentNet scaffold files.",
    )
    init_parser.set_defaults(handler=_handle_init)

    doctor_parser = subcommands.add_parser("doctor", help="Check the local AgentNet environment.")
    doctor_parser.add_argument("--artifact", help="Optional .agentnet artifact to validate.")
    doctor_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    doctor_parser.set_defaults(handler=_handle_doctor)

    inspect_parser = subcommands.add_parser("inspect", help="Inspect a .agentnet artifact.")
    inspect_parser.add_argument("artifact", help="Artifact directory to inspect.")
    inspect_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    inspect_parser.set_defaults(handler=_handle_inspect)

    export_parser = subcommands.add_parser(
        "export",
        help="Export a .agentnet artifact as a Python package.",
    )
    export_parser.add_argument("artifact", help="Artifact directory to export.")
    export_parser.add_argument("--package", required=True, help="Generated package name.")
    export_parser.add_argument(
        "--output",
        required=True,
        help="Generated package output directory.",
    )
    export_parser.add_argument("--module-name", help="Generated import module name.")
    export_parser.add_argument("--version", default="0.1.0", help="Generated package version.")
    export_parser.add_argument("--description", help="Generated package description.")
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory.",
    )
    export_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    export_parser.set_defaults(handler=_handle_export)

    eval_parser = subcommands.add_parser(
        "eval",
        help="Evaluate descriptor prediction cases against an artifact.",
    )
    eval_parser.add_argument("artifact", help="Artifact directory to validate before evaluation.")
    eval_parser.add_argument("--cases", required=True, help="JSON cases file.")
    eval_parser.add_argument("--output", required=True, help="Evaluation report JSON path.")
    eval_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    eval_parser.set_defaults(handler=_handle_eval)

    train_parser = subcommands.add_parser(
        "train",
        help="Write descriptor training history from a JSON dataset.",
    )
    train_parser.add_argument("artifact", help="Artifact directory to validate before training.")
    train_parser.add_argument("--dataset", required=True, help="Training dataset JSON path.")
    train_parser.add_argument("--output", required=True, help="Training history report JSON path.")
    train_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    train_parser.set_defaults(handler=_handle_train)

    return parser


def _handle_init(args: argparse.Namespace) -> int:
    project_dir = Path(args.path)
    project_name = str(args.name or project_dir.name or "agentnet-project")
    module_name = _module_name(project_name)

    if project_dir.exists() and any(project_dir.iterdir()) and not args.force:
        raise ValueError(f"Project directory {str(project_dir)!r} already exists and is not empty")

    files = {
        project_dir / "README.md": _render_project_readme(project_name, module_name),
        project_dir / "agentnet.toml": _render_project_config(project_name, module_name),
        project_dir / "examples" / "basic_agent.py": _render_basic_agent_example(),
        project_dir / "src" / module_name / "__init__.py": (
            f'"""AgentNet project package for {project_name}."""\n'
        ),
        project_dir / "data" / "eval_cases.json": _render_eval_cases(),
    }
    for path, content in files.items():
        if path.exists() and not args.force:
            raise ValueError(f"File {str(path)!r} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Created AgentNet project at {project_dir}")
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "agentnet_version": __version__,
        "checks": {
            "package_importable": True,
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "status": "ok",
    }
    if args.artifact is not None:
        result = validate_artifact(args.artifact)
        payload["artifact"] = result.to_dict()
        if not result.passed:
            payload["status"] = "error"

    _emit_payload(
        payload,
        json_output=bool(args.json),
        text=f"AgentNet doctor: {payload['status']}",
    )
    return 0 if payload["status"] == "ok" else 1


def _handle_inspect(args: argparse.Namespace) -> int:
    payload = _artifact_summary(Path(args.artifact))
    _emit_payload(
        payload,
        json_output=bool(args.json),
        text=(
            f"Artifact {payload['manifest']['name']} "
            f"({payload['graph']['type']}): "
            f"{'valid' if payload['validation']['passed'] else 'invalid'}"
        ),
    )
    return 0 if payload["validation"]["passed"] else 1


def _handle_export(args: argparse.Namespace) -> int:
    result = export_package(
        args.artifact,
        args.output,
        package_name=args.package,
        module_name=args.module_name,
        version=args.version,
        description=args.description,
        overwrite=bool(args.overwrite),
    )
    payload = result.to_dict()
    _emit_payload(
        payload,
        json_output=bool(args.json),
        text=f"Exported {result.package_name} to {result.path}",
    )
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    summary = _artifact_summary(Path(args.artifact))
    if not summary["validation"]["passed"]:
        raise ValueError("Artifact validation failed")

    cases = _case_items(_read_json(Path(args.cases)))
    results = [_score_case(index, case) for index, case in enumerate(cases, start=1)]
    score = sum(result.score for result in results) / len(results) if results else 0.0
    passed = bool(results) and all(result.passed for result in results)
    payload = {
        "artifact": summary["manifest"],
        "case_count": len(results),
        "passed": passed,
        "results": [result.to_dict() for result in results],
        "score": score,
    }
    validate_descriptor_payload_no_secrets(payload, label="CLI evaluation report")
    _write_json(Path(args.output), payload)
    _emit_payload(
        payload,
        json_output=bool(args.json),
        text=f"Wrote evaluation report to {args.output}",
    )
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    summary = _artifact_summary(Path(args.artifact))
    if not summary["validation"]["passed"]:
        raise ValueError("Artifact validation failed")

    dataset_payload = _read_json(Path(args.dataset))
    dataset = Dataset.from_dict(_dataset_payload(dataset_payload))
    raw_examples = _case_items(dataset_payload)
    steps = [
        _training_step(index, example, raw_examples[index - 1])
        for index, example in enumerate(dataset, start=1)
    ]
    history = TrainingHistory(
        steps,
        metadata={
            "artifact_name": summary["manifest"]["name"],
            "mode": "descriptor",
            "source": "agentnet.cli.train",
        },
    )
    payload = {
        "artifact": summary["manifest"],
        "best_score": history.best_score,
        "dataset": dataset.to_dict(),
        "example_count": len(dataset),
        "history": history.to_dict(),
    }
    validate_descriptor_payload_no_secrets(payload, label="CLI training report")
    _write_json(Path(args.output), payload)
    _emit_payload(
        payload,
        json_output=bool(args.json),
        text=f"Wrote training history to {args.output}",
    )
    return 0


def _artifact_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path / "manifest.json")
    graph = _read_json(path / "graph.json")
    validation = validate_artifact(path)
    agents = _agent_descriptors(graph)
    return {
        "dependencies": {
            "llms": sorted(
                {
                    str(model_ref["alias"])
                    for agent in agents
                    for model_ref in agent.get("llms", ())
                }
            ),
            "mcp_tools": sorted(
                {
                    str(tool)
                    for agent in agents
                    for tool in agent.get("tools", ())
                    if "." in str(tool)
                }
            ),
            "tools": sorted(
                {
                    str(tool)
                    for agent in agents
                    for tool in agent.get("tools", ())
                    if "." not in str(tool)
                }
            ),
        },
        "graph": {
            "agent_count": len(agents),
            "type": str(graph.get("type")),
        },
        "manifest": {
            "agentnet_version": str(manifest["agentnet_version"]),
            "artifact_version": str(manifest["artifact_version"]),
            "name": str(manifest["name"]),
        },
        "path": str(path),
        "validation": validation.to_dict(),
    }


def _agent_descriptors(descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
    module_type = descriptor.get("type")
    if module_type == "ReActAgent":
        return [dict(descriptor)]
    if module_type in {"Sequential", "Parallel"}:
        agents: list[dict[str, Any]] = []
        for child in descriptor.get("modules", ()):
            if isinstance(child, Mapping):
                agents.extend(_agent_descriptors(child))
        reducer = descriptor.get("reducer")
        if isinstance(reducer, Mapping):
            agents.extend(_agent_descriptors(reducer))
        return agents
    if module_type == "Router":
        agents = []
        router = descriptor.get("router")
        if isinstance(router, Mapping):
            agents.extend(_agent_descriptors(router))
        for route in dict(descriptor.get("routes", {})).values():
            if isinstance(route, Mapping):
                agents.extend(_agent_descriptors(route))
        fallback = descriptor.get("fallback")
        if isinstance(fallback, Mapping):
            agents.extend(_agent_descriptors(fallback))
        return agents
    if module_type == "Reducer":
        reducer = descriptor.get("reducer")
        return [] if not isinstance(reducer, Mapping) else _agent_descriptors(reducer)
    if module_type == "DAG":
        agents = []
        for node in dict(descriptor.get("nodes", {})).values():
            if isinstance(node, Mapping):
                agents.extend(_agent_descriptors(node))
        return agents
    return []


def _score_case(index: int, case: Mapping[str, Any]) -> EvaluationResult:
    case_id = str(case.get("id", f"case-{index}"))
    expected = case.get("expected", case.get("expected_output"))
    actual = case.get("actual", case.get("actual_output"))
    if "actual" not in case and "actual_output" not in case:
        failure = EvaluationFailure(
            objective="exact_match",
            message="Case is missing actual output.",
            metadata={"case_id": case_id},
        )
        return EvaluationResult(
            score=0.0,
            passed=False,
            failures=(failure,),
            metadata={"case_id": case_id},
        )
    passed = actual == expected
    failures = ()
    if not passed:
        failures = (
            EvaluationFailure(
                objective="exact_match",
                message="Actual output did not match expected output.",
                metadata={"case_id": case_id},
            ),
        )
    return EvaluationResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        failures=failures,
        metrics={"exact_match": 1.0 if passed else 0.0},
        metadata={"case_id": case_id},
    )


def _training_step(
    index: int,
    example: Any,
    raw_example: Mapping[str, Any],
) -> TrainingStep:
    score = float(raw_example.get("score", 0.0))
    passed = bool(raw_example.get("passed", score >= 1.0))
    example_id = example.id or str(raw_example.get("id", f"case-{index}"))
    return TrainingStep(
        epoch=1,
        example_id=example_id,
        passed=passed,
        score=score,
        metrics={"cli.train.score": score},
        metadata={"source": "agentnet.cli.train"},
    )


def _dataset_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"examples": payload, "name": None}
    if isinstance(payload, Mapping):
        if "examples" in payload:
            return dict(payload)
        if "cases" in payload:
            return {
                "examples": payload["cases"],
                "metadata": dict(payload.get("metadata", {})),
                "name": payload.get("name"),
            }
    raise ValueError("Dataset JSON must be a list or an object with examples")


def _case_items(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        items = payload.get("cases", payload.get("examples"))
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("Cases JSON must be a list or an object with cases/examples")
    if not all(isinstance(item, Mapping) for item in items):
        raise ValueError("Cases must be objects")
    return [dict(item) for item in items]


def _module_name(project_name: str) -> str:
    module_name = project_name.strip().lower().replace("-", "_").replace(".", "_")
    module_name = re.sub(r"[^a-z0-9_]", "_", module_name)
    if not module_name or module_name[0].isdigit():
        module_name = f"agentnet_{module_name}"
    if _MODULE_NAME_PATTERN.fullmatch(module_name) is None:
        raise ValueError(f"Cannot derive a Python module name from {project_name!r}")
    return module_name


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit_payload(payload: Mapping[str, Any], *, json_output: bool, text: str) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(text)


def _render_project_config(project_name: str, module_name: str) -> str:
    return (
        "[project]\n"
        f'name = "{project_name}"\n'
        f'module = "{module_name}"\n'
        'artifact = "artifacts/model.agentnet"\n'
        "\n"
        "[runtime]\n"
        'llms = []\n'
        "tools = []\n"
    )


def _render_project_readme(project_name: str, module_name: str) -> str:
    return (
        f"# {project_name}\n"
        "\n"
        "AgentNet project scaffold.\n"
        "\n"
        "```bash\n"
        "agentnet doctor\n"
        "agentnet inspect artifacts/model.agentnet\n"
        "```\n"
        "\n"
        f"Python package module: `{module_name}`.\n"
    )


def _render_basic_agent_example() -> str:
    return (
        "import agentnet as an\n"
        "\n"
        "agent = an.ReActAgent(\n"
        '    "planner",\n'
        '    instructions="Plan the work clearly.",\n'
        '    llms=["strong"],\n'
        ")\n"
        "\n"
        "# Inject a real LLM backend at runtime before executing the agent.\n"
    )


def _render_eval_cases() -> str:
    return json.dumps(
        [
            {
                "actual": "example output",
                "expected": "example output",
                "id": "case-1",
                "input": "example input",
            }
        ],
        indent=2,
        sort_keys=True,
    ) + "\n"
