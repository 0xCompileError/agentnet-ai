"""Descriptor-only AgentNet artifact serialization."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import reduce
from hashlib import sha256
from operator import or_
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from agentnet._version import __version__
from agentnet.agents import ReActAgent
from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetValidationError,
    Module,
    Schema,
)
from agentnet.graphs import DAG, Parallel, Reducer, Router, Sequential, validate_graph
from agentnet.llms import ModelRef
from agentnet.mcp import MCPRegistry
from agentnet.mcp._security import (
    validate_descriptor_payload_no_secrets,
    validate_safe_metadata,
)
from agentnet.policies import RetryPolicy
from agentnet.tools import ToolRegistry
from agentnet.training import TrainingHistory

ARTIFACT_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class ArtifactVersion:
    """Artifact format version with compatibility checks."""

    value: str

    def __post_init__(self) -> None:
        _parse_version(self.value)

    def is_compatible_with(self, current: str = ARTIFACT_VERSION) -> bool:
        major, minor = _parse_version(self.value)
        current_major, current_minor = _parse_version(current)
        return major == current_major and minor <= current_minor


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Top-level artifact manifest."""

    artifact_version: str
    agentnet_version: str
    name: str
    created_at: str
    graph_hash: str
    prompt_hash: str
    schema_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise AgentNetConfigurationError("ArtifactManifest name cannot be empty")
        ArtifactVersion(self.artifact_version)
        metadata = dict(self.metadata)
        validate_safe_metadata(metadata, label="ArtifactManifest")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        graph: Mapping[str, Any],
        prompts: Mapping[str, str] | None = None,
        schemas: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        return cls(
            artifact_version=ARTIFACT_VERSION,
            agentnet_version=__version__,
            name=name,
            created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
            graph_hash=_hash_payload(dict(graph)),
            prompt_hash=_hash_payload(dict(prompts or {})),
            schema_hash=_hash_payload(dict(schemas or {})),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentnet_version": self.agentnet_version,
            "artifact_version": self.artifact_version,
            "created_at": self.created_at,
            "graph_hash": self.graph_hash,
            "metadata": self.metadata.copy(),
            "name": self.name,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, manifest: Mapping[str, Any]) -> ArtifactManifest:
        return cls(
            artifact_version=str(manifest["artifact_version"]),
            agentnet_version=str(manifest["agentnet_version"]),
            name=str(manifest["name"]),
            created_at=str(manifest["created_at"]),
            graph_hash=str(manifest["graph_hash"]),
            prompt_hash=str(manifest["prompt_hash"]),
            schema_hash=str(manifest["schema_hash"]),
            metadata=dict(manifest.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ArtifactValidationResult:
    """Validation report for an artifact directory."""

    passed: bool
    failures: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "failures", tuple(dict(failure) for failure in self.failures))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "failures": [dict(failure) for failure in self.failures],
            "metadata": self.metadata.copy(),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class AgentNetArtifact:
    """In-memory view of a saved artifact."""

    path: Path
    manifest: ArtifactManifest
    graph: dict[str, Any]
    prompts: dict[str, str]
    schemas: dict[str, Any]
    tools: dict[str, Any]
    mcp: dict[str, Any]
    training_history: dict[str, Any] | None = None


def save(
    module: Module,
    path: str | Path,
    *,
    name: str | None = None,
    tools: ToolRegistry | None = None,
    mcp_registry: MCPRegistry | None = None,
    training_history: TrainingHistory | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = True,
) -> AgentNetArtifact:
    """Save a module graph as a descriptor-only ``.agentnet`` directory."""

    if not isinstance(module, Module):
        raise AgentNetConfigurationError("save requires a Module")
    artifact_path = Path(path)
    artifact_name = name or artifact_path.stem
    graph = serialize_graph(module)
    prompts = _collect_prompts(graph)
    schemas = _collect_schemas(graph)
    tools_payload = tools.to_dict() if tools is not None else {"tools": []}
    mcp_payload = mcp_registry.to_dict() if mcp_registry is not None else {"servers": []}
    training_payload = (
        None if training_history is None else training_history.to_dict()
    )
    manifest = ArtifactManifest.create(
        name=artifact_name,
        graph=graph,
        prompts=prompts,
        schemas=schemas,
        metadata=metadata,
    )

    artifact_payload = {
        "graph": graph,
        "manifest": manifest.to_dict(),
        "mcp": mcp_payload,
        "prompts": prompts,
        "schemas": schemas,
        "tools": tools_payload,
        "training_history": training_payload,
    }
    validate_descriptor_payload_no_secrets(artifact_payload, label="AgentNet artifact")

    if artifact_path.exists():
        if not overwrite:
            raise AgentNetConfigurationError(
                f"Artifact path {str(artifact_path)!r} already exists"
            )
        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
        else:
            artifact_path.unlink()
    _write_artifact(
        artifact_path,
        manifest=manifest,
        graph=graph,
        prompts=prompts,
        schemas=schemas,
        tools=tools_payload,
        mcp=mcp_payload,
        training_history=training_payload,
    )
    return AgentNetArtifact(
        path=artifact_path,
        manifest=manifest,
        graph=graph,
        prompts=prompts,
        schemas=schemas,
        tools=tools_payload,
        mcp=mcp_payload,
        training_history=training_payload,
    )


def load(
    path: str | Path,
    *,
    llms: Mapping[str, Any] | None = None,
    tools: Mapping[str, Callable[..., Any]] | ToolRegistry | None = None,
    mcp_servers: Mapping[str, Any] | MCPRegistry | None = None,
    tracer: Any | None = None,
    scheduler: Any | None = None,
) -> Module:
    """Load an artifact with explicitly injected runtime dependencies."""

    del tracer, scheduler
    artifact_path = Path(path)
    result = validate_artifact(
        artifact_path,
        llms=dict(llms or {}),
        tools={} if tools is None else tools,
        mcp_servers=MCPRegistry() if mcp_servers is None else mcp_servers,
    )
    if not result.passed:
        messages = "; ".join(str(failure["message"]) for failure in result.failures)
        raise AgentNetValidationError(messages)
    graph = _read_json(artifact_path / "graph.json")
    prompts = _read_prompts(artifact_path)
    schemas = _read_schemas(artifact_path)
    return _deserialize_module(
        graph,
        llms=dict(llms or {}),
        prompts=prompts,
        schemas=schemas,
    )


def validate_artifact(
    path: str | Path,
    *,
    llms: Mapping[str, Any] | None = None,
    tools: Mapping[str, Callable[..., Any]] | ToolRegistry | None = None,
    mcp_servers: Mapping[str, Any] | MCPRegistry | None = None,
) -> ArtifactValidationResult:
    """Validate artifact structure, version, hashes, and optional dependencies."""

    artifact_path = Path(path)
    failures: list[dict[str, Any]] = []
    if not artifact_path.is_dir():
        return ArtifactValidationResult(
            passed=False,
            failures=(
                {
                    "code": "missing_artifact",
                    "message": f"Artifact path {str(artifact_path)!r} is not a directory",
                },
            ),
        )

    try:
        manifest = ArtifactManifest.from_dict(_read_json(artifact_path / "manifest.json"))
    except Exception as exc:
        return ArtifactValidationResult(
            passed=False,
            failures=(
                {
                    "code": "invalid_manifest",
                    "message": str(exc),
                },
            ),
        )

    graph = _read_optional_json(artifact_path / "graph.json", failures, "missing_graph")
    if not ArtifactVersion(manifest.artifact_version).is_compatible_with():
        failures.append(
            {
                "code": "incompatible_version",
                "message": (
                    f"Artifact version {manifest.artifact_version!r} is not compatible "
                    f"with {ARTIFACT_VERSION!r}"
                ),
            }
        )

    if graph is not None and _hash_payload(graph) != manifest.graph_hash:
        failures.append(
            {
                "code": "hash_mismatch",
                "message": "Graph hash does not match manifest",
            }
        )

    prompts = _read_prompts(artifact_path)
    if _hash_payload(prompts) != manifest.prompt_hash:
        failures.append(
            {
                "code": "hash_mismatch",
                "message": "Prompt hash does not match manifest",
            }
        )

    schemas = _read_schemas(artifact_path)
    if _hash_payload(schemas) != manifest.schema_hash:
        failures.append(
            {
                "code": "hash_mismatch",
                "message": "Schema hash does not match manifest",
            }
        )

    if graph is not None:
        _validate_schema_descriptors(schemas, failures)
        if llms is not None:
            missing_aliases = sorted(_required_model_aliases(graph) - set(llms))
            if missing_aliases:
                failures.append(
                    {
                        "code": "missing_llms",
                        "message": f"Missing LLM aliases: {', '.join(missing_aliases)}",
                    }
                )
        if tools is not None:
            missing_tools = sorted(_required_tool_names(graph) - _provided_tool_names(tools))
            if missing_tools:
                failures.append(
                    {
                        "code": "missing_tools",
                        "message": f"Missing tools: {', '.join(missing_tools)}",
                    }
                )
        if mcp_servers is not None:
            registry = _coerce_mcp_registry(mcp_servers)
            missing_mcp_tools = sorted(
                tool
                for tool in _required_mcp_tool_names(graph)
                if not registry.is_tool_allowed(tool)
            )
            if missing_mcp_tools:
                failures.append(
                    {
                        "code": "missing_mcp_tools",
                        "message": f"Missing MCP tools: {', '.join(missing_mcp_tools)}",
                    }
                )

    return ArtifactValidationResult(
        passed=not failures,
        failures=tuple(failures),
        metadata={
            "artifact_version": manifest.artifact_version,
            "name": manifest.name,
        },
    )


def serialize_graph(module: Module) -> dict[str, Any]:
    """Serialize a supported module graph into a safe descriptor."""

    validate_graph(module)
    descriptor = _serialize_module(module)
    validate_descriptor_payload_no_secrets(descriptor, label="Graph descriptor")
    return descriptor


def serialize_schema(schema: Any) -> dict[str, Any]:
    """Serialize a supported schema object or type annotation."""

    if isinstance(schema, Schema):
        return {
            "fields": {
                field: _serialize_annotation(annotation)
                for field, annotation in sorted(schema.fields.items())
            },
            "type": "Schema",
        }
    return _serialize_annotation(schema)


def deserialize_schema(descriptor: Mapping[str, Any]) -> Any:
    """Deserialize a schema descriptor without loading executable code."""

    descriptor_type = descriptor.get("type")
    if descriptor_type == "Schema":
        return Schema(
            {
                str(field): _deserialize_annotation(dict(annotation))
                for field, annotation in dict(descriptor.get("fields", {})).items()
            }
        )
    return _deserialize_annotation(descriptor)


def _serialize_module(module: Module) -> dict[str, Any]:
    if isinstance(module, ReActAgent):
        descriptor: dict[str, Any] = {
            "input_schema_ref": None,
            "instructions": module.instructions,
            "llms": [_serialize_llm_ref(llm) for llm in module.llms],
            "max_steps": module.max_steps,
            "metadata": module.metadata.copy(),
            "name": module.name,
            "output_schema_ref": None,
            "prompt_ref": None,
            "retry_policy": (
                None
                if module.retry_policy is None or not hasattr(module.retry_policy, "to_dict")
                else module.retry_policy.to_dict()
            ),
            "tools": list(module.tools),
            "type": "ReActAgent",
        }
        if module.instructions is not None:
            descriptor["prompt_ref"] = f"prompts/{module.name}.md"
        if module.input_schema is not None:
            descriptor["input_schema_ref"] = f"schemas/{module.name}.input_schema.json"
            descriptor["input_schema"] = serialize_schema(module.input_schema)
        output_schema = module.output_schema
        if output_schema is not None:
            descriptor["output_schema_ref"] = f"schemas/{module.name}.output_schema.json"
            descriptor["output_schema"] = serialize_schema(output_schema)
        return descriptor

    if isinstance(module, Sequential):
        return {
            "modules": [_serialize_module(child) for child in module.modules],
            "name": module.name,
            "type": "Sequential",
        }

    if isinstance(module, Parallel):
        return {
            "modules": [_serialize_module(child) for child in module.modules],
            "name": module.name,
            "reducer": (
                None if module.reducer is None else _serialize_module(module.reducer)
            ),
            "type": "Parallel",
        }

    if isinstance(module, Router):
        return {
            "fallback": (
                None if module.fallback is None else _serialize_module(module.fallback)
            ),
            "name": module.name,
            "router": _serialize_module(module.router),
            "routes": {
                route_name: _serialize_module(route)
                for route_name, route in sorted(module.routes.items())
            },
            "type": "Router",
        }

    if isinstance(module, Reducer):
        return {
            "name": module.name,
            "reducer": _serialize_module(module.reducer),
            "type": "Reducer",
        }

    if isinstance(module, DAG):
        return {
            "edges": {source: list(targets) for source, targets in module.edges.items()},
            "name": module.name,
            "nodes": {
                node_name: _serialize_module(child)
                for node_name, child in sorted(module.nodes.items())
            },
            "type": "DAG",
        }

    raise AgentNetConfigurationError(
        f"Cannot serialize unsupported module type {module.__class__.__name__!r}"
    )


def _deserialize_module(
    descriptor: Mapping[str, Any],
    *,
    llms: Mapping[str, Any],
    prompts: Mapping[str, str],
    schemas: Mapping[str, Any],
) -> Module:
    module_type = str(descriptor["type"])
    if module_type == "ReActAgent":
        model_backends = [
            llms[model_ref["alias"]]
            for model_ref in descriptor.get("llms", ())
        ]
        input_schema_ref = descriptor.get("input_schema_ref")
        output_schema_ref = descriptor.get("output_schema_ref")
        retry_policy = descriptor.get("retry_policy")
        prompt_ref = descriptor.get("prompt_ref")
        prompt = None if prompt_ref is None else prompts[str(prompt_ref)]
        return ReActAgent(
            str(descriptor["name"]),
            instructions=prompt,
            llms=model_backends,
            tools=tuple(descriptor.get("tools", ())),
            retry_policy=(
                None if retry_policy is None else RetryPolicy.from_dict(dict(retry_policy))
            ),
            input_schema=(
                None
                if input_schema_ref is None
                else deserialize_schema(schemas[str(input_schema_ref)])
            ),
            output_schema=(
                None
                if output_schema_ref is None
                else deserialize_schema(schemas[str(output_schema_ref)])
            ),
            max_steps=int(descriptor.get("max_steps", 8)),
            metadata=dict(descriptor.get("metadata", {})),
        )

    if module_type == "Sequential":
        return Sequential(
            *[
                _deserialize_module(child, llms=llms, prompts=prompts, schemas=schemas)
                for child in descriptor.get("modules", ())
            ],
            name=str(descriptor["name"]),
        )

    if module_type == "Parallel":
        reducer = descriptor.get("reducer")
        return Parallel(
            *[
                _deserialize_module(child, llms=llms, prompts=prompts, schemas=schemas)
                for child in descriptor.get("modules", ())
            ],
            reducer=(
                None
                if reducer is None
                else _deserialize_module(reducer, llms=llms, prompts=prompts, schemas=schemas)
            ),
            name=str(descriptor["name"]),
        )

    if module_type == "Router":
        fallback = descriptor.get("fallback")
        return Router(
            router=_deserialize_module(
                dict(descriptor["router"]),
                llms=llms,
                prompts=prompts,
                schemas=schemas,
            ),
            routes={
                str(name): _deserialize_module(route, llms=llms, prompts=prompts, schemas=schemas)
                for name, route in dict(descriptor.get("routes", {})).items()
            },
            fallback=(
                None
                if fallback is None
                else _deserialize_module(fallback, llms=llms, prompts=prompts, schemas=schemas)
            ),
            name=str(descriptor["name"]),
        )

    if module_type == "Reducer":
        return Reducer(
            _deserialize_module(
                dict(descriptor["reducer"]),
                llms=llms,
                prompts=prompts,
                schemas=schemas,
            ),
            name=str(descriptor["name"]),
        )

    if module_type == "DAG":
        return DAG(
            nodes={
                str(name): _deserialize_module(node, llms=llms, prompts=prompts, schemas=schemas)
                for name, node in dict(descriptor.get("nodes", {})).items()
            },
            edges={
                str(source): [str(target) for target in targets]
                for source, targets in dict(descriptor.get("edges", {})).items()
            },
            name=str(descriptor["name"]),
        )

    raise AgentNetValidationError(f"Unsupported artifact module type {module_type!r}")


def _write_artifact(
    path: Path,
    *,
    manifest: ArtifactManifest,
    graph: Mapping[str, Any],
    prompts: Mapping[str, str],
    schemas: Mapping[str, Any],
    tools: Mapping[str, Any],
    mcp: Mapping[str, Any],
    training_history: Mapping[str, Any] | None,
) -> None:
    path.mkdir(parents=True)
    for subdir in ("agents", "prompts", "schemas", "tools", "mcp", "training"):
        (path / subdir).mkdir()

    _write_json(path / "manifest.json", manifest.to_dict())
    _write_json(path / "graph.json", graph)
    for agent in _agent_descriptors(graph):
        _write_json(path / "agents" / f"{agent['name']}.json", agent)
    for prompt_ref, prompt in prompts.items():
        (path / prompt_ref).write_text(prompt, encoding="utf-8")
    for schema_ref, schema in schemas.items():
        _write_json(path / schema_ref, schema)
    _write_json(path / "tools" / "manifest.json", tools)
    _write_json(path / "mcp" / "manifest.json", mcp)
    if training_history is not None:
        _write_json(path / "training" / "history.json", training_history)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(
    path: Path,
    failures: list[dict[str, Any]],
    code: str,
) -> dict[str, Any] | None:
    try:
        return _read_json(path)
    except FileNotFoundError:
        failures.append({"code": code, "message": f"Missing {path.name}"})
        return None


def _collect_prompts(descriptor: Mapping[str, Any]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for agent in _agent_descriptors(descriptor):
        prompt_ref = agent.get("prompt_ref")
        if prompt_ref is not None:
            prompts[str(prompt_ref)] = str(agent.get("_prompt", ""))
    return prompts


def _collect_schemas(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    for agent in _agent_descriptors(descriptor):
        input_schema_ref = agent.get("input_schema_ref")
        if input_schema_ref is not None:
            schemas[str(input_schema_ref)] = agent["input_schema"]
        output_schema_ref = agent.get("output_schema_ref")
        if output_schema_ref is not None:
            schemas[str(output_schema_ref)] = agent["output_schema"]
    return schemas


def _read_prompts(path: Path) -> dict[str, str]:
    prompt_dir = path / "prompts"
    if not prompt_dir.is_dir():
        return {}
    return {
        f"prompts/{prompt_path.name}": prompt_path.read_text(encoding="utf-8")
        for prompt_path in sorted(prompt_dir.glob("*.md"))
    }


def _read_schemas(path: Path) -> dict[str, Any]:
    schema_dir = path / "schemas"
    if not schema_dir.is_dir():
        return {}
    return {
        f"schemas/{schema_path.name}": _read_json(schema_path)
        for schema_path in sorted(schema_dir.glob("*.json"))
    }


def _agent_descriptors(descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
    module_type = descriptor.get("type")
    if module_type == "ReActAgent":
        agent = dict(descriptor)
        prompt = descriptor.get("instructions")
        if prompt is not None:
            agent["_prompt"] = prompt
        return [agent]
    if module_type in {"Sequential", "Parallel"}:
        agents: list[dict[str, Any]] = []
        for child in descriptor.get("modules", ()):
            agents.extend(_agent_descriptors(child))
        reducer = descriptor.get("reducer")
        if reducer is not None:
            agents.extend(_agent_descriptors(reducer))
        return agents
    if module_type == "Router":
        agents = _agent_descriptors(dict(descriptor["router"]))
        for route in dict(descriptor.get("routes", {})).values():
            agents.extend(_agent_descriptors(route))
        fallback = descriptor.get("fallback")
        if fallback is not None:
            agents.extend(_agent_descriptors(fallback))
        return agents
    if module_type == "Reducer":
        return _agent_descriptors(dict(descriptor["reducer"]))
    if module_type == "DAG":
        agents: list[dict[str, Any]] = []
        for node in dict(descriptor.get("nodes", {})).values():
            agents.extend(_agent_descriptors(node))
        return agents
    return []


def _serialize_llm_ref(llm: Any) -> dict[str, Any]:
    if isinstance(llm, ModelRef):
        return llm.to_dict()
    if hasattr(llm, "name") and hasattr(llm, "model"):
        return ModelRef(
            alias=str(llm.name),
            provider=llm.__class__.__name__,
            model=str(llm.model),
        ).to_dict()
    raise AgentNetConfigurationError("ReActAgent llms must be model refs or backends")


def _serialize_annotation(annotation: Any) -> dict[str, Any]:
    simple_types = {
        str: "str",
        int: "int",
        float: "float",
        bool: "bool",
        list: "list",
        dict: "dict",
        Any: "any",
        type(None): "none",
        None: "none",
    }
    if annotation in simple_types:
        return {"type": simple_types[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        item = args[0] if args else Any
        return {"items": _serialize_annotation(item), "type": "list"}
    if origin is dict:
        key = args[0] if len(args) >= 1 else str
        value = args[1] if len(args) >= 2 else Any
        return {
            "keys": _serialize_annotation(key),
            "type": "dict",
            "values": _serialize_annotation(value),
        }
    if origin in (Union, UnionType):
        return {"type": "union", "options": [_serialize_annotation(option) for option in args]}

    raise AgentNetConfigurationError(f"Unsupported schema annotation: {annotation!r}")


def _deserialize_annotation(descriptor: Mapping[str, Any]) -> Any:
    descriptor_type = descriptor.get("type")
    if descriptor_type == "str":
        return str
    if descriptor_type == "int":
        return int
    if descriptor_type == "float":
        return float
    if descriptor_type == "bool":
        return bool
    if descriptor_type == "list":
        item = descriptor.get("items")
        return list if item is None else list[_deserialize_annotation(dict(item))]
    if descriptor_type == "dict":
        key = descriptor.get("keys")
        value = descriptor.get("values")
        if key is None or value is None:
            return dict
        return dict[_deserialize_annotation(dict(key)), _deserialize_annotation(dict(value))]
    if descriptor_type == "any":
        return Any
    if descriptor_type == "none":
        return type(None)
    if descriptor_type == "union":
        options = tuple(
            _deserialize_annotation(dict(option))
            for option in descriptor.get("options", ())
        )
        if not options:
            raise AgentNetConfigurationError("Union schema descriptor requires options")
        return reduce(or_, options)
    raise AgentNetConfigurationError(f"Unsupported schema descriptor type {descriptor_type!r}")


def _validate_schema_descriptors(
    schemas: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    for schema_ref, schema in schemas.items():
        try:
            deserialize_schema(schema)
        except Exception as exc:
            failures.append(
                {
                    "code": "invalid_schema",
                    "message": f"Invalid schema {schema_ref}: {exc}",
                }
            )


def _required_model_aliases(descriptor: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for agent in _agent_descriptors(descriptor):
        aliases.update(str(model_ref["alias"]) for model_ref in agent.get("llms", ()))
    return aliases


def _required_tool_names(descriptor: Mapping[str, Any]) -> set[str]:
    return {
        tool
        for agent in _agent_descriptors(descriptor)
        for tool in (str(name) for name in agent.get("tools", ()))
        if "." not in tool
    }


def _required_mcp_tool_names(descriptor: Mapping[str, Any]) -> set[str]:
    return {
        tool
        for agent in _agent_descriptors(descriptor)
        for tool in (str(name) for name in agent.get("tools", ()))
        if "." in tool
    }


def _provided_tool_names(tools: Mapping[str, Callable[..., Any]] | ToolRegistry) -> set[str]:
    if isinstance(tools, ToolRegistry):
        return set(tools.names)
    return set(str(name) for name in tools)


def _coerce_mcp_registry(mcp_servers: Mapping[str, Any] | MCPRegistry) -> MCPRegistry:
    if isinstance(mcp_servers, MCPRegistry):
        return mcp_servers
    registry = MCPRegistry()
    for server in mcp_servers.values():
        registry.register(server)
    return registry


def _hash_payload(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(data).hexdigest()}"


def _parse_version(version: str) -> tuple[int, int]:
    major, separator, minor = version.partition(".")
    if not separator:
        raise AgentNetConfigurationError("Artifact version must be '<major>.<minor>'")
    return int(major), int(minor)
