import json
from pathlib import Path
from typing import Any

import pytest

import agentnet as an


def search_docs(query: str) -> list[str]:
    return [query]


def test_artifact_manifest_hashes_and_round_trips() -> None:
    manifest = an.ArtifactManifest.create(
        name="decision_net",
        graph={"type": "ReActAgent", "name": "planner"},
        prompts={"planner": "Plan clearly."},
        schemas={"planner.output": {"type": "Schema", "fields": {}}},
    )

    serialized = manifest.to_dict()

    assert serialized["artifact_version"] == an.ARTIFACT_VERSION
    assert serialized["agentnet_version"] == an.__version__
    assert serialized["graph_hash"].startswith("sha256:")
    assert serialized["prompt_hash"].startswith("sha256:")
    assert serialized["schema_hash"].startswith("sha256:")
    assert an.ArtifactManifest.from_dict(serialized).to_dict() == serialized


def test_save_artifact_writes_directory_layout_without_live_secrets(
    tmp_path: Path,
) -> None:
    registry = an.ToolRegistry()
    registry.register("search_docs", search_docs, description="Search docs")
    mcp_registry = an.MCPRegistry()
    mcp_registry.register(
        an.MCPServer(
            name="github",
            command=["npx", "server-github"],
            env={"GITHUB_TOKEN": "not serialized"},
            tools=[an.MCPToolDescriptor("search_repos")],
        ),
        allow_tools=["search_repos"],
    )
    history = an.TrainingHistory(
        [
            an.TrainingStep(
                epoch=1,
                example_id="case-1",
                score=0.75,
                passed=True,
            )
        ]
    )
    net = an.Sequential(
        an.ReActAgent(
            "planner",
            instructions="Plan clearly.",
            llms=["strong"],
            tools=["search_docs", "github.search_repos"],
            output_schema=an.Schema({"summary": str}),
        ),
        name="decision_net",
    )
    artifact_path = tmp_path / "decision_net.agentnet"

    saved = an.save(
        net,
        artifact_path,
        name="decision_net",
        tools=registry,
        mcp_registry=mcp_registry,
        training_history=history,
    )

    assert saved.path == artifact_path
    assert (artifact_path / "manifest.json").is_file()
    assert (artifact_path / "graph.json").is_file()
    assert (artifact_path / "agents" / "planner.json").is_file()
    assert (artifact_path / "prompts" / "planner.md").read_text() == "Plan clearly."
    assert (artifact_path / "schemas" / "planner.output_schema.json").is_file()
    assert (artifact_path / "tools" / "manifest.json").is_file()
    assert (artifact_path / "mcp" / "manifest.json").is_file()
    assert (artifact_path / "training" / "history.json").is_file()
    serialized_artifact = json.dumps(
        json.loads((artifact_path / "manifest.json").read_text()),
        sort_keys=True,
    )
    assert "GITHUB_TOKEN" not in serialized_artifact
    assert "not serialized" not in serialized_artifact


def test_load_artifact_rehydrates_graph_with_injected_dependencies(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "planner.agentnet"
    an.save(
        an.ReActAgent(
            "planner",
            instructions="Plan clearly.",
            llms=["strong"],
        ),
        artifact_path,
        name="planner",
    )
    llm = an.FakeLLM(responses=["final answer"], name="strong")

    loaded = an.load(artifact_path, llms={"strong": llm})

    assert isinstance(loaded, an.ReActAgent)
    assert loaded.instructions == "Plan clearly."
    assert loaded.llms == (llm,)
    assert an.run(loaded, "input") == "final answer"


def test_load_artifact_validates_required_model_aliases(tmp_path: Path) -> None:
    artifact_path = tmp_path / "planner.agentnet"
    an.save(an.ReActAgent("planner", llms=["strong"]), artifact_path, name="planner")

    with pytest.raises(an.AgentNetValidationError, match="Missing LLM aliases"):
        an.load(artifact_path, llms={})


def test_load_artifact_validates_required_tools_and_mcp_allowlists(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "planner.agentnet"
    an.save(
        an.ReActAgent(
            "planner",
            llms=["strong"],
            tools=["search_docs", "github.search_repos"],
        ),
        artifact_path,
        name="planner",
        mcp_registry=an.MCPRegistry(
            [
                an.MCPServer(
                    name="github",
                    command=["npx", "server-github"],
                    tools=[an.MCPToolDescriptor("search_repos")],
                )
            ]
        ),
    )

    with pytest.raises(an.AgentNetValidationError, match="Missing tools"):
        an.load(artifact_path, llms={"strong": an.FakeLLM(name="strong")}, tools={})

    mcp_registry = an.MCPRegistry(
        [
            an.MCPServer(
                name="github",
                command=["npx", "server-github"],
                tools=[an.MCPToolDescriptor("search_repos")],
            )
        ]
    )
    loaded = an.load(
        artifact_path,
        llms={"strong": an.FakeLLM(name="strong")},
        tools={"search_docs": search_docs},
        mcp_servers=mcp_registry,
    )

    assert isinstance(loaded, an.ReActAgent)


def test_validate_artifact_detects_incompatible_versions(tmp_path: Path) -> None:
    artifact_path = tmp_path / "planner.agentnet"
    an.save(an.ReActAgent("planner", llms=[]), artifact_path, name="planner")
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest))

    result = an.validate_artifact(artifact_path)

    assert result.passed is False
    assert result.failures[0]["code"] == "incompatible_version"


def test_save_artifact_rejects_secret_like_payload_keys(tmp_path: Path) -> None:
    with pytest.raises(an.AgentNetConfigurationError, match="may serialize secrets"):
        an.save(
            an.ReActAgent(
                "planner",
                llms=[],
                metadata={"api_token": "secret"},
            ),
            tmp_path / "planner.agentnet",
            name="planner",
        )


def test_artifact_validation_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "planner.agentnet"
    an.save(an.ReActAgent("planner", llms=[]), artifact_path, name="planner")
    graph_path = artifact_path / "graph.json"
    graph = json.loads(graph_path.read_text())
    graph["name"] = "tampered"
    graph_path.write_text(json.dumps(graph))

    result = an.validate_artifact(artifact_path)

    assert result.passed is False
    assert result.failures[0]["code"] == "hash_mismatch"


def test_schema_descriptor_round_trips_supported_annotations() -> None:
    schema = an.Schema({"summary": str, "scores": list[int]})

    descriptor = an.serialize_schema(schema)
    restored = an.deserialize_schema(descriptor)

    assert descriptor == {
        "fields": {
            "scores": {"items": {"type": "int"}, "type": "list"},
            "summary": {"type": "str"},
        },
        "type": "Schema",
    }
    assert isinstance(restored, an.Schema)
    assert restored.validate({"summary": "ok", "scores": [1]}) == {
        "scores": [1],
        "summary": "ok",
    }


def test_artifact_public_exports_are_available() -> None:
    exported: list[Any] = [
        an.ARTIFACT_VERSION,
        an.AgentNetArtifact,
        an.ArtifactManifest,
        an.ArtifactValidationResult,
        an.ArtifactVersion,
        an.deserialize_schema,
        an.load,
        an.save,
        an.serialize_schema,
        an.validate_artifact,
    ]

    assert all(value is not None for value in exported)
