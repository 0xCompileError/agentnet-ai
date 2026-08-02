from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import agentnet as an

ROOT = Path(__file__).resolve().parents[1]


class FakeLangSmithClient:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def create_run(self, **payload: Any) -> None:
        self.runs.append(payload)


class EchoModule(an.Module):
    async def arun(self, input: Any, context: Any | None = None) -> Any:
        del context
        return f"echo:{input}"


def search_docs(query: str) -> dict[str, list[str]]:
    return {"matches": [f"doc:{query}"]}


def _pythonpath_with_repo_src(extra: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(ROOT / "src")]
    if extra is not None:
        entries.insert(0, str(extra))
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _import_from_src(module_name: str, source_root: Path) -> Any:
    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(source_root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(source_root))


def test_release_candidate_docs_cover_public_api_and_release_checks() -> None:
    api_reference = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs" / "release-candidate.md").read_text(encoding="utf-8")

    for phrase in (
        "# AgentNet API Reference",
        "ReActAgent",
        "Sequential",
        "Trainer",
        "TopologyOptimizer",
        "PluginManager",
        "LangSmithExporter",
        "MCPToolAdapter",
        "export_package",
    ):
        assert phrase in api_reference

    for phrase in (
        "uv run ruff check .",
        "uv run pyright",
        "uv run pytest",
        "uv build",
        "benchmarks/release_candidate.py",
        "TestPyPI",
        "PyPI",
        "credentials",
    ):
        assert phrase in runbook

    token_pattern = re.compile(r"\bpypi-[A-Za-z0-9_=-]{12,}")
    assert token_pattern.search(api_reference) is None
    assert token_pattern.search(runbook) is None


def test_release_candidate_benchmark_script_runs_without_secrets() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "release_candidate.py")],
        capture_output=True,
        check=True,
        env=_pythonpath_with_repo_src(),
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["benchmark"] == "release_candidate"
    assert payload["ok"] is True
    assert payload["iterations"] >= 5
    assert payload["agent_runs"] == payload["iterations"]
    assert payload["elapsed_ms"] >= 0
    assert "pypi-" not in completed.stdout.lower()
    assert "token" not in completed.stdout.lower()
    assert completed.stderr == ""


def test_release_candidate_local_integration_flow(tmp_path: Path) -> None:
    tools = an.ToolRegistry()
    tools.register(
        "search_docs",
        search_docs,
        description="Search release documentation.",
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"matches": list}),
    )

    fake_mcp = an.FakeMCPServer(name="docs")
    fake_mcp.register_tool(
        "search",
        lambda query: {"matches": [query]},
        input_schema=an.Schema({"query": str}),
        output_schema=an.Schema({"matches": list}),
    )
    mcp_registry = fake_mcp.to_registry(allow_tools=["search"])

    net = an.ReActAgent(
        "planner",
        instructions="Plan clearly.",
        llms=["strong"],
        tools=["search_docs", "docs.search"],
    )
    artifact_path = tmp_path / "rc_net.agentnet"
    an.save(
        net,
        artifact_path,
        name="rc_net",
        tools=tools,
        mcp_registry=mcp_registry,
        metadata={"stage": "release-candidate"},
    )

    validation = an.validate_artifact(
        artifact_path,
        llms={"strong": an.FakeLLM(name="strong", responses=["validated"])},
        tools=tools,
        mcp_servers=mcp_registry,
    )
    assert validation.passed is True

    loaded = an.load(
        artifact_path,
        llms={"strong": an.FakeLLM(name="strong", responses=["loaded"])},
        tools={"search_docs": search_docs},
        mcp_servers=mcp_registry,
    )
    assert an.run(loaded, "release input") == "loaded"

    exported = an.export_package(
        artifact_path,
        tmp_path / "rc_package",
        package_name="rc-net",
        overwrite=True,
    )
    rc_net = _import_from_src("rc_net", exported.path / "src")
    package_validation = rc_net.validate(
        llms={"strong": an.FakeLLM(name="strong", responses=["package validate"])},
        tools=tools,
        mcp_servers=mcp_registry,
    )
    assert package_validation.passed is True
    package_loaded = rc_net.load(
        llms={"strong": an.FakeLLM(name="strong", responses=["package loaded"])},
        tools={"search_docs": search_docs},
        mcp_servers=mcp_registry,
    )
    assert an.run(package_loaded, "release input") == "package loaded"

    client = FakeLangSmithClient()
    span = an.TraceSpan.start(
        "planner",
        run_id="rc-run",
        kind="agent",
        attributes={"agent": "planner"},
    )
    span.finish()
    payloads = an.LangSmithExporter(client=client, project_name="agentnet-rc").export(
        an.Trace(run_id="rc-run", spans=[span])
    )
    assert client.runs == payloads
    assert payloads[0]["project_name"] == "agentnet-rc"

    approvals = an.MCPApprovalStore()
    approvals.approve("docs.search", mcp_registry.get_tool("docs.search"))
    adapter = an.MCPToolAdapter(
        mcp_registry,
        fake_mcp,
        approvals=approvals,
        require_approval=True,
    )
    adapted_tools = an.ToolRegistry()
    adapter.register_all(adapted_tools)
    mcp_context = an.RunContext("rc-mcp")
    mcp_result = adapted_tools.execute(
        "docs.search",
        {"query": "agentnet"},
        agent=an.ReActAgent("researcher", tools=["docs.search"]),
        context=mcp_context,
    )
    assert mcp_result == {"matches": ["agentnet"]}
    assert len(mcp_context.metadata["mcp_events"]) == 2

    scheduler_context = an.RunContext("rc-scheduler")
    scheduler = an.ThreadPoolScheduler(max_workers=1)
    assert an.run(
        EchoModule("echo"),
        "payload",
        scheduler=scheduler,
        context=scheduler_context,
    ) == "echo:payload"
    assert scheduler_context.metadata["scheduler_events"][0]["type"] == "scheduler.submit"
