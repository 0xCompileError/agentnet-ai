# AgentNet API Reference

This reference summarizes the public v0.1 API exported from `agentnet`.
AgentNet uses descriptor-only artifacts and explicit dependency injection:
callers provide live LLMs, tools, MCP clients, tracers, schedulers, and
application callables at runtime.

## Runtime Core

- `Module`: base class for executable AgentNet components.
- `run` and `arun`: sync and async package-level execution helpers.
- `RunContext`: per-run metadata, cancellation, tracing, graph state, and
  runtime constraint carrier.
- `AgentState`, `GraphState`, and `GraphResult`: serializable runtime state and
  result records.
- `Schema`: lightweight runtime schema validator for inputs, outputs, tools,
  and artifact descriptors.
- `AgentNetError`, `AgentNetConfigurationError`, `AgentNetExecutionError`,
  `AgentNetStateError`, and `AgentNetValidationError`: public error hierarchy.

```python
import agentnet as an

agent = an.ReActAgent("planner", llms=[an.FakeLLM(responses=["ok"])])
result = an.run(agent, "input", context=an.RunContext("run-1"))
```

## Agents And Graphs

- `ReActAgent`: core computational unit with instructions, LLM fallback policy,
  schemas, retry policy, tool allowlists, and metadata.
- `Sequential`: executes modules in order.
- `Parallel`: executes branches and optionally applies a reducer.
- `Router`: selects a route through a router module and fallback policy.
- `Reducer`: applies a reducer module to aggregate upstream values.
- `DAG`: executes named modules according to dependency edges.
- `build_shape`: builds common sequential and parallel shapes from tuples.
- `compile_graph`, `validate_graph`, and `visualize_graph`: graph inspection
  helpers.
- `GraphCompiler`, `GraphValidator`, and `GraphVisualizer`: reusable graph
  service classes.

```python
net = an.Sequential(
    an.ReActAgent("planner", llms=["strong"]),
    an.ReActAgent("writer", llms=["cheap"]),
)
```

## LLM Layer

- `LLMBackend`: runtime-checkable provider protocol.
- `FakeLLM`: deterministic backend for tests, examples, and benchmarks.
- `LiteLLM`, `OpenAI`, `Anthropic`, `Bedrock`, `VertexAI`, and
  `OpenAICompatible`: provider adapters.
- `ModelRef`: descriptor-safe model alias reference.
- `LLMPolicy`: ordered primary and fallback model selection.
- `ChatRequest`, `ChatResponse`, and `ChatEvent`: provider request, response,
  and streaming event records.

## Tools And MCP

- `ToolSpec`: descriptor-only tool metadata, schemas, and side-effect flags.
- `ToolRegistry`: explicit in-process tool registration and execution.
- `MCPServer` and `MCPToolDescriptor`: descriptor-only MCP server and tool
  records.
- `MCPRegistry`: allowlisted MCP descriptor registry.
- `MCPToolAdapter`: adapts approved MCP descriptors into `ToolRegistry`
  callables using an explicitly injected MCP client.
- `FakeMCPServer`: deterministic in-memory MCP server for tests and local
  examples.
- `MCPApprovalStore`, `MCPDescriptorApproval`, `hash_mcp_descriptor`, and
  `validate_mcp_descriptor`: approval and descriptor-control utilities.

## Constraints And Interfaces

- `Constraint`, `ConstraintResult`, `ConstraintDescriptor`, and
  `ConstraintKind`: base constraint model.
- `AndConstraint`, `OrConstraint`, `NotConstraint`, and `CompositeConstraint`:
  constraint composition.
- `NodeConstraint`, `EdgeConstraint`, `GraphConstraint`, and `GraphEdge`: scoped
  constraints.
- Built-ins: `SchemaConstraint`, `RepresentationConstraint`,
  `ModelConstraint`, `ToolConstraint`, `RetryConstraint`, `CostConstraint`,
  `TokenConstraint`, `LatencyConstraint`, `TopologyConstraint`,
  `MemoryConstraint`, `SafetyConstraint`, and `CustomConstraint`.
- `Interface`, `Representation`, `SemanticContract`, and descriptor records:
  communication contract model.
- Built-in representations: JSON Schema, Pydantic-style model, Markdown, plain
  text, bullet list, XML, YAML, key-value, and evidence graph.
- Validators and detectors: `validate_interface_compatibility`,
  `negotiate_representation`, `select_representation`,
  `validate_semantic_equivalence`, `validate_information_preservation`,
  `detect_lossy_translation`, and `detect_incompatible_interfaces`.

## Evaluation And Training

- `train` and `atrain`: beginner-facing end-to-end training from ordinary `X`
  and `y` values, with inferred scoring, deterministic validation, and bounded
  optimization.
- `FittedAgentNet`: runnable selected network with `run`, `arun`, `evaluate`,
  `aevaluate`, `save`, and a `.training` report.
- `TrainingReport`, `TrainingTrial`, and `TrainingTrialEvent`: descriptor-safe
  optimization provenance and progress.
- `AutoOptimizer` and `ExplicitCandidates`: default staged optimization and
  explicit candidate selection through the same workflow.
- `Objective`, `ObjectiveSuite`, `EvaluationResult`,
  `EvaluationFailure`, and `aggregate_evaluation_results`: evaluation framework.
- Built-in objectives: `SchemaObjective`, `JudgeObjective`,
  `ExactMatchObjective`, `ExpectedOutputObjective`, `HumanFeedbackObjective`,
  `UnitTestObjective`,
  `CostObjective`, `LatencyObjective`, `ToolEfficiencyObjective`,
  `CustomObjective`, and `CustomMetricObjective`.
- `Dataset` and `TrainingExample`: training data records.
- `Trainer`, `TrainingResult`, and `TrainingCandidateResult`: eval-driven fitting
  with ordered per-candidate records and explicit tie reporting.
- `TrainingProgressEvent`: descriptor-safe training, candidate, and example
  lifecycle callbacks.
- `TrainingCheckpoint`, `TrainingHistory`, and `TrainingStep`: descriptor-only
  training history.
- `Budget` and `BudgetManager`: epoch, example, trial, estimated LLM-call, and
  cost budget tracking.
- `AttributionEngine`, `AttributionRecord`, `TrainingPatch`, and
  `generate_training_patch`: patch attribution and reversible patch utilities.

## Optimizers

- `PromptOptimizer`, `PromptOptimizationResult`, and
  `ConstraintAwareOptimizer`: prompt and constraint-aware candidate selection.
- `FallbackOptimizer` and `RetryPolicyOptimizer`: training policy optimizers.
- `TopologySearchSpace`, `TopologyMutation`, `TopologyMutationEngine`,
  `TopologyCandidate`, `ArchitectureScore`, `ArchitectureScorer`,
  `TopologyOptimizer`, `TopologyOptimizationResult`, and
  `TopologyCheckpoint`: topology search.
- `RepresentationSelectionOptimizer`, `CommunicationProtocolOptimizer`,
  `InterfaceCompatibilityOptimizer`, `TranslationStrategyOptimizer`, and
  `InformationTransferOptimizer`: communication and representation optimizers.

## Serialization And Export

- `save`, `load`, and `validate_artifact`: `.agentnet` artifact lifecycle.
- `AgentNetArtifact`, `ArtifactManifest`, `ArtifactValidationResult`,
  `ArtifactVersion`, and `ARTIFACT_VERSION`: artifact records and versioning.
- `serialize_schema` and `deserialize_schema`: schema descriptor helpers.
- `PackageExporter`, `PackageExportResult`, and `export_package`: generated
  package export for validated artifacts.

```python
artifact = an.save(net, "decision_net.agentnet", name="decision_net")
exported = an.export_package(artifact.path, "dist/decision_net", package_name="decision-net")
```

## Runtime Schedulers

- `Scheduler`: scheduler protocol.
- `NodeSpec`, `NodeFuture`, and `NodeResult`: descriptor-safe scheduling
  records.
- `LocalScheduler`, `ThreadPoolScheduler`, and `ProcessPoolScheduler`: local
  and standard-library execution backends.
- `RayScheduler`, `CeleryScheduler`, and `TemporalScheduler`: dependency-light
  adapters over explicitly injected clients.
- `RetryPolicy`: transport and quality retry policy with fallback and backoff
  configuration.

## Tracing And Observability

- `TraceEvent`, `TraceSpan`, `Trace`, and `TraceMetrics`: normalized tracing
  records.
- `InMemoryTracer`: local trace recorder.
- `trace_from_context`: builds a normalized trace from runtime metadata.
- `LangSmithExporter`: adapts traces to a LangSmith-compatible injected client.
- `OpenTelemetryExporter`: adapts traces to OpenTelemetry-style span payloads.
- `record_topology_result`: records topology search checkpoints and best
  candidate events.

## Plugins

- `PluginDescriptor` and `PluginKind`: descriptor-only plugin metadata.
- `PluginRegistry`: base descriptor/factory registry.
- `PluginManager`: central registry container.
- Category registries: `ProviderPluginRegistry`, `OptimizerPluginRegistry`,
  `EvaluatorPluginRegistry`, `SchedulerPluginRegistry`,
  `TracerPluginRegistry`, `StoragePluginRegistry`, and
  `MemoryPluginRegistry`.
