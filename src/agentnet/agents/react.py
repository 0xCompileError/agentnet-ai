"""ReAct agent configuration shell."""

import time
from collections.abc import Mapping, Sequence
from typing import Any

import anyio

from agentnet.agents.messages import MessageBuilder
from agentnet.core import (
    AgentNetConfigurationError,
    AgentNetExecutionError,
    AgentNetValidationError,
    AgentState,
    Module,
)
from agentnet.interfaces import Interface
from agentnet.llms import ChatRequest, ModelRef
from agentnet.tracing import record_llm_event


class ReActAgent(Module):
    """Configurable ReAct agent node."""

    def __init__(
        self,
        name: str,
        instructions: str | None = None,
        llms: Sequence[Any] | None = None,
        tools: Sequence[str] | None = None,
        retry_policy: Any | None = None,
        input_schema: Any | None = None,
        output_schema: Any | None = None,
        interface: Interface | None = None,
        input_interface: Interface | None = None,
        max_steps: int = 8,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self.instructions = instructions
        self.llms = tuple(_normalize_llm(llm) for llm in llms or ())
        self.tools = tuple(tools or ())
        self.retry_policy = retry_policy
        self.input_schema = input_schema
        self.input_interface = _normalize_interface(
            input_interface,
            input_schema,
            interface_name="input_interface",
            schema_name="input_schema",
        )
        self.interface = _normalize_interface(
            interface,
            output_schema,
            interface_name="interface",
            schema_name="output_schema",
        )
        self.output_schema = output_schema
        self.max_steps = max_steps
        self.metadata = dict(metadata or {})

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return self.tools

    def allows_tool(self, name: str) -> bool:
        return name in self.tools

    def require_tool(self, name: str) -> str:
        if not self.allows_tool(name):
            raise AgentNetExecutionError(
                f"Tool {name!r} is not allowed for ReActAgent {self.name!r}"
            )
        return name

    async def arun(self, input: Any, context: Any | None = None) -> Any:
        _validate_interface(self.input_interface, input, label="input")
        agent_state = _context_agent_state(context, self.name)
        if agent_state is not None and agent_state.step >= self.max_steps:
            raise AgentNetExecutionError(
                f"ReActAgent {self.name!r} reached max steps ({self.max_steps})"
            )

        messages = MessageBuilder(self.instructions).build(input, state=agent_state)
        response = await _complete_with_model_switching(
            self.llms,
            messages,
            self.retry_policy,
            self.interface,
            agent_name=self.name,
            context=context,
        )
        if agent_state is not None:
            agent_state.add_reasoning(response.content, model=str(response.model))
            agent_state.advance()
        return response.content

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "instructions": self.instructions,
                "llms": [_serialize_llm(llm) for llm in self.llms],
                "max_steps": self.max_steps,
                "metadata": self.metadata.copy(),
                "tools": list(self.tools),
            }
        )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.instructions = state.get("instructions")
        self.llms = tuple(ModelRef.from_dict(llm) for llm in state.get("llms", []))
        self.max_steps = int(state.get("max_steps", 8))
        self.metadata = dict(state.get("metadata", {}))
        self.tools = tuple(state.get("tools", ()))
        self.retry_policy = None
        self.input_schema = None
        self.interface = None
        self.input_interface = None
        self.output_schema = None


def _normalize_llm(llm: Any) -> Any:
    if isinstance(llm, str):
        return ModelRef(llm)
    return llm


def _normalize_interface(
    interface: Interface | None,
    schema: Any | None,
    *,
    interface_name: str,
    schema_name: str,
) -> Interface | None:
    if interface is not None and schema is not None:
        raise AgentNetConfigurationError(
            f"ReActAgent accepts either {interface_name} or {schema_name}, not both"
        )
    if interface is not None:
        return interface
    if schema is not None:
        return Interface(schema=schema)
    return None


async def _complete_with_model_switching(
    llms: tuple[Any, ...],
    messages: Sequence[Mapping[str, str]],
    retry_policy: Any | None,
    interface: Interface | None,
    *,
    agent_name: str,
    context: Any | None,
) -> Any:
    live_llms = _live_llms(llms)
    last_error: AgentNetExecutionError | AgentNetValidationError | None = None

    for index, llm in enumerate(live_llms):
        request = ChatRequest(model=str(llm.name), messages=messages)
        try:
            return await _complete_with_quality_retries(
                llm,
                request,
                retry_policy,
                interface,
                agent_name=agent_name,
                context=context,
            )
        except (AgentNetExecutionError, AgentNetValidationError) as exc:
            last_error = exc
            fallback_reason = _fallback_reason(exc)
            if index == len(live_llms) - 1:
                raise
            if not _should_switch_model(retry_policy, fallback_reason):
                raise

    if last_error is not None:
        raise last_error
    raise AgentNetConfigurationError("ReActAgent requires a live LLM backend to execute")


def _live_llms(llms: tuple[Any, ...]) -> tuple[Any, ...]:
    live_llms = tuple(
        llm for llm in llms if not isinstance(llm, ModelRef) and hasattr(llm, "complete")
    )
    if live_llms:
        return live_llms
    raise AgentNetConfigurationError("ReActAgent requires a live LLM backend to execute")


def _fallback_reason(error: AgentNetExecutionError | AgentNetValidationError) -> str:
    if isinstance(error, AgentNetValidationError):
        return "schema_failure"

    cause = error.__cause__
    if isinstance(cause, TimeoutError):
        return "timeout"
    if cause is not None and "RateLimit" in type(cause).__name__:
        return "rate_limit"
    return "api_error"


def _should_switch_model(retry_policy: Any | None, reason: str) -> bool:
    if retry_policy is None or not hasattr(retry_policy, "should_fallback"):
        return False
    return bool(retry_policy.should_fallback(reason))


async def _complete_with_quality_retries(
    llm: Any,
    request: ChatRequest,
    retry_policy: Any | None,
    interface: Interface | None,
    *,
    agent_name: str,
    context: Any | None,
) -> Any:
    attempts_used = 0
    quality_attempts = _quality_attempt_limit(retry_policy)
    last_validation_error: AgentNetValidationError | None = None

    for quality_attempt in range(1, quality_attempts + 1):
        remaining_attempts = _remaining_total_attempts(retry_policy, attempts_used)
        if remaining_attempts == 0:
            break

        response, transport_attempts = await _complete_with_transport_retries(
            llm,
            request,
            retry_policy,
            max_attempts=remaining_attempts,
            agent_name=agent_name,
            context=context,
        )
        attempts_used += transport_attempts

        try:
            _validate_interface(interface, response.content, label="output")
        except AgentNetValidationError as exc:
            last_validation_error = exc
            if quality_attempt == quality_attempts:
                raise
            if _remaining_total_attempts(retry_policy, attempts_used) == 0:
                raise
            delay_seconds = _retry_delay(retry_policy, quality_attempt)
            _record_retry_event(
                context,
                agent_name=agent_name,
                attempt=quality_attempt,
                delay_seconds=delay_seconds,
                error=exc,
                model=str(response.model),
                reason="quality",
            )
            await _sleep_for_delay(delay_seconds)
            continue

        return response

    if last_validation_error is not None:
        raise last_validation_error
    raise AgentNetExecutionError("ReActAgent LLM attempt budget was exhausted")


def _validate_interface(
    interface: Interface | None,
    value: Any,
    *,
    label: str,
) -> Any:
    if interface is None:
        return value
    return interface.validate(value, label=label)


async def _complete_with_transport_retries(
    llm: Any,
    request: ChatRequest,
    retry_policy: Any | None,
    max_attempts: int | None = None,
    *,
    agent_name: str,
    context: Any | None,
) -> tuple[Any, int]:
    attempts = _transport_attempt_limit(retry_policy)
    if max_attempts is not None:
        attempts = min(attempts, max_attempts)

    for attempt in range(1, attempts + 1):
        started_ns = time.perf_counter_ns()
        try:
            response = await llm.complete(request)
            _record_llm_completion_event(
                context,
                agent_name=agent_name,
                attempt=attempt,
                duration_ms=_elapsed_ms(started_ns),
                response=response,
            )
            return response, attempt
        except Exception as exc:
            record_llm_event(
                context,
                agent_name=agent_name,
                attempt=attempt,
                duration_ms=_elapsed_ms(started_ns),
                error=exc,
                event_type="llm.failed",
                model=str(request.model),
            )
            if attempt == attempts:
                raise AgentNetExecutionError(
                    f"ReActAgent LLM transport failure after {attempts} attempt(s)"
                ) from exc
            delay_seconds = _retry_delay(retry_policy, attempt)
            _record_retry_event(
                context,
                agent_name=agent_name,
                attempt=attempt,
                delay_seconds=delay_seconds,
                error=exc,
                model=str(request.model),
                reason="transport",
            )
            await _sleep_for_delay(delay_seconds)

    raise AgentNetExecutionError("ReActAgent LLM transport failure")


def _record_llm_completion_event(
    context: Any | None,
    *,
    agent_name: str,
    attempt: int,
    duration_ms: float,
    response: Any,
) -> None:
    metadata = getattr(response, "metadata", {})
    cost_usd = _extract_cost_usd(metadata)
    usage = getattr(response, "usage", {})
    record_llm_event(
        context,
        agent_name=agent_name,
        attempt=attempt,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        event_type="llm.completed",
        model=str(getattr(response, "model", "")),
        usage=usage if isinstance(usage, Mapping) else None,
    )


def _extract_cost_usd(metadata: Any) -> float | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in ("cost_usd", "total_cost_usd", "cost"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    return None


def _elapsed_ms(started_ns: int) -> float:
    return max(0.000001, (time.perf_counter_ns() - started_ns) / 1_000_000)


def _retry_delay(retry_policy: Any | None, retry_number: int) -> float:
    if retry_policy is None or not hasattr(retry_policy, "backoff_delay"):
        return 0.0

    return float(retry_policy.backoff_delay(retry_number))


async def _sleep_for_delay(delay_seconds: float) -> None:
    if delay_seconds > 0:
        await anyio.sleep(delay_seconds)


def _record_retry_event(
    context: Any | None,
    *,
    agent_name: str,
    attempt: int,
    delay_seconds: float,
    error: Exception,
    model: str,
    reason: str,
) -> None:
    if context is None or not hasattr(context, "metadata"):
        return

    retry_events = context.metadata.setdefault("retry_events", [])
    if not isinstance(retry_events, list):
        return

    retry_events.append(
        {
            "agent": agent_name,
            "attempt": attempt,
            "delay_seconds": delay_seconds,
            "error_type": type(error).__name__,
            "model": model,
            "next_attempt": attempt + 1,
            "reason": reason,
            "type": "retry.started",
        }
    )
    _record_retry_metrics(context, reason=reason, delay_seconds=delay_seconds)


def _record_retry_metrics(
    context: Any,
    *,
    reason: str,
    delay_seconds: float,
) -> None:
    metrics = context.metadata.setdefault(
        "retry_metrics",
        {
            "quality_retries": 0,
            "total_backoff_seconds": 0.0,
            "total_retries": 0,
            "transport_retries": 0,
        },
    )
    if not isinstance(metrics, dict):
        return

    reason_key = f"{reason}_retries"
    metrics["total_retries"] = int(metrics.get("total_retries", 0)) + 1
    metrics[reason_key] = int(metrics.get(reason_key, 0)) + 1
    metrics["total_backoff_seconds"] = (
        float(metrics.get("total_backoff_seconds", 0.0)) + delay_seconds
    )


def _quality_attempt_limit(retry_policy: Any | None) -> int:
    if retry_policy is None:
        return 1
    return max(1, 1 + int(getattr(retry_policy, "quality_retries", 0)))


def _transport_attempt_limit(retry_policy: Any | None) -> int:
    if retry_policy is None:
        return 1

    attempts = 1 + int(getattr(retry_policy, "transport_retries", 0))
    max_total_attempts = getattr(retry_policy, "max_total_attempts", None)
    if max_total_attempts is not None:
        attempts = min(attempts, int(max_total_attempts))
    return max(1, attempts)


def _remaining_total_attempts(retry_policy: Any | None, attempts_used: int) -> int | None:
    if retry_policy is None:
        return None

    max_total_attempts = getattr(retry_policy, "max_total_attempts", None)
    if max_total_attempts is None:
        return None
    return max(0, int(max_total_attempts) - attempts_used)


def _context_agent_state(context: Any | None, name: str) -> AgentState | None:
    if context is None or not hasattr(context, "graph_state"):
        return None

    graph_state = context.graph_state
    try:
        return graph_state.get_agent_state(name)
    except KeyError:
        agent_state = AgentState(name=name)
        graph_state.set_agent_state(agent_state)
        return agent_state


def _serialize_llm(llm: Any) -> dict[str, Any]:
    if isinstance(llm, ModelRef):
        return llm.to_dict()
    if hasattr(llm, "name") and hasattr(llm, "model"):
        return ModelRef(
            alias=str(llm.name),
            provider=type(llm).__name__,
            model=str(llm.model),
        ).to_dict()
    raise AgentNetConfigurationError("ReActAgent llms must be model refs or backends")
