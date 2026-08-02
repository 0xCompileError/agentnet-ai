from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import agentnet as an

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-m", "agentnet", *args],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


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
    )
    return artifact_path


def test_cli_help_and_project_script_are_available() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    result = _run_cli("--help")

    assert result.returncode == 0
    assert "agentnet init" in result.stdout
    assert "doctor" in result.stdout
    assert pyproject["project"]["scripts"]["agentnet"] == "agentnet.cli:main"


def test_agentnet_init_creates_project_scaffold(tmp_path: Path) -> None:
    project_dir = tmp_path / "support-bot"

    result = _run_cli("init", str(project_dir), "--name", "support-bot")

    assert result.returncode == 0
    assert (project_dir / "agentnet.toml").is_file()
    assert (project_dir / "src" / "support_bot" / "__init__.py").is_file()
    assert (project_dir / "examples" / "basic_agent.py").is_file()
    config = (project_dir / "agentnet.toml").read_text()
    assert 'name = "support-bot"' in config
    assert 'module = "support_bot"' in config


def test_agentnet_doctor_reports_json_status_and_artifact_validation(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path)

    result = _run_cli("doctor", "--artifact", str(artifact_path), "--json")

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["checks"]["package_importable"] is True
    assert payload["artifact"]["passed"] is True


def test_agentnet_inspect_reports_artifact_summary(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path)

    result = _run_cli("inspect", str(artifact_path), "--json")

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["manifest"]["name"] == "decision_net"
    assert payload["graph"]["type"] == "ReActAgent"
    assert payload["dependencies"]["llms"] == ["strong"]
    assert payload["validation"]["passed"] is True


def test_agentnet_export_generates_package_from_cli(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path)
    output_dir = tmp_path / "decision_package"

    result = _run_cli(
        "export",
        str(artifact_path),
        "--package",
        "decision-net",
        "--output",
        str(output_dir),
        "--json",
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["package_name"] == "decision-net"
    assert (output_dir / "pyproject.toml").is_file()
    assert (output_dir / "src" / "decision_net" / "_loader.py").is_file()


def test_agentnet_eval_scores_prediction_cases(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path)
    cases_path = tmp_path / "cases.json"
    report_path = tmp_path / "eval-report.json"
    cases_path.write_text(
        json.dumps(
            [
                {"id": "case-1", "actual": "yes", "expected": "yes"},
                {"id": "case-2", "actual": "no", "expected": "yes"},
            ]
        )
    )

    result = _run_cli(
        "eval",
        str(artifact_path),
        "--cases",
        str(cases_path),
        "--output",
        str(report_path),
    )

    payload = json.loads(report_path.read_text())
    assert result.returncode == 0
    assert payload["case_count"] == 2
    assert payload["score"] == 0.5
    assert payload["passed"] is False


def test_agentnet_train_writes_descriptor_training_history(tmp_path: Path) -> None:
    artifact_path = _make_artifact(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    history_path = tmp_path / "training-history.json"
    dataset_path.write_text(
        json.dumps(
            {
                "examples": [
                    {"id": "case-1", "input": "a", "expected_output": "A", "score": 1.0},
                    {"id": "case-2", "input": "b", "expected_output": "B", "score": 0.25},
                ],
                "name": "cli-dataset",
            }
        )
    )

    result = _run_cli(
        "train",
        str(artifact_path),
        "--dataset",
        str(dataset_path),
        "--output",
        str(history_path),
    )

    payload = json.loads(history_path.read_text())
    assert result.returncode == 0
    assert payload["example_count"] == 2
    assert payload["history"]["steps"][0]["example_id"] == "case-1"
    assert payload["history"]["steps"][0]["score"] == 1.0
    assert payload["best_score"] == 1.0
