from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = (
    ("01_single_agent", "single_agent", "Single Agent"),
    ("02_sequential_pipeline", "sequential_pipeline", "Sequential Pipeline"),
    (
        "03_parallel_research_pipeline",
        "parallel_research_pipeline",
        "Parallel Research Pipeline",
    ),
    ("04_router", "router", "Router"),
    ("05_mixture_of_experts", "mixture_of_experts", "Mixture Of Experts"),
    ("06_litellm_gateway", "litellm_gateway", "LiteLLM Gateway"),
    ("07_mcp", "mcp", "MCP"),
    ("08_langsmith", "langsmith", "LangSmith"),
    ("09_distributed_execution", "distributed_execution", "Distributed Execution"),
    ("10_topology_search", "topology_search", "Topology Search"),
    ("11_exported_package", "exported_package", "Exported Package"),
    ("12_training_10_examples", "training_10_examples", "Training 10 Examples"),
)


def test_milestone_examples_have_readmes_and_runnable_scripts() -> None:
    for directory, slug, title in EXAMPLES:
        example_dir = ROOT / "examples" / directory
        readme = example_dir / "README.md"
        script = example_dir / "main.py"

        assert readme.is_file(), f"{directory} should include README.md"
        assert script.is_file(), f"{directory} should include main.py"

        readme_text = readme.read_text()
        assert f"# {title}" in readme_text
        assert f"python examples/{directory}/main.py" in readme_text

        payload = _run_example(script)

        assert payload["example"] == slug
        assert payload["ok"] is True
        assert payload["result"]


def _run_example(script: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["AGENTNET_TRAINING_LLM"] = "fake"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
    )
    stdout = completed.stdout.strip()
    assert stdout, f"{script} produced no output"
    assert "api_key" not in stdout.lower()
    assert "password" not in stdout.lower()
    assert "secret" not in stdout.lower()
    payload = json.loads(stdout)
    assert isinstance(payload, dict)
    return payload


def test_training_example_uses_ten_examples_and_selects_best_candidate() -> None:
    payload = _run_example(ROOT / "examples" / "12_training_10_examples" / "main.py")
    result = payload["result"]

    assert isinstance(result, dict)
    assert result["dataset_size"] == 10
    assert result["evaluated_candidates"] == 2
    assert result["best_candidate"] == "triage_candidate"
    assert result["checkpoint_count"] == 2
    assert result["history_steps"] == 10
    assert result["is_tied"] is False
    assert result["llm_mode"] == "fake"
    assert result["score"] == 1.0
    assert result["training_examples"] == 10.0
    assert result["passed"] is True
    assert result["candidates"] == [
        {
            "name": "triage_baseline",
            "passed": False,
            "prompt": (
                "Classify each support ticket as billing, bug, or how_to. "
                "Return exactly one label and no other text."
            ),
            "score": 0.6,
        },
        {
            "name": "triage_candidate",
            "passed": True,
            "prompt": (
                "Classify each support ticket using only these labels: billing, "
                "bug, how_to. Return exactly one lowercase label and no punctuation."
            ),
            "score": 1.0,
        },
    ]


class FakeOpenAIResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": self.content},
                }
            ],
            "usage": {"completion_tokens": 1, "prompt_tokens": 10},
        }


class FakeOpenAIClient:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeOpenAIResponse:
        self.requests.append({"headers": headers, "json": json, "url": url})
        return FakeOpenAIResponse(self.responses[len(self.requests) - 1])


def test_training_example_openai_mode_uses_injected_http_client_offline() -> None:
    script = ROOT / "examples" / "12_training_10_examples" / "main.py"
    namespace = runpy.run_path(str(script))
    cases = cast(tuple[tuple[str, str], ...], namespace["CASES"])
    baseline_labels = cast(tuple[str, ...], namespace["BASELINE_LABELS"])
    baseline_prompt = cast(str, namespace["BASELINE_PROMPT"])
    candidate_prompt = cast(str, namespace["CANDIDATE_PROMPT"])
    run_training = cast(Any, namespace["run_training"])
    responses = (*baseline_labels, *(label for _, label in cases))
    client = FakeOpenAIClient(responses)

    result = run_training(
        llm_mode="openai",
        openai_client=client,
        api_key="offline-credential",
        model="offline-training-model",
        progress_callback=lambda event: None,
    )

    assert result["best_candidate"] == "triage_candidate"
    assert result["score"] == 1.0
    assert len(client.requests) == 20
    assert {
        request["json"]["model"] for request in client.requests
    } == {"offline-training-model"}
    assert {
        request["url"] for request in client.requests
    } == {"https://api.openai.com/v1/chat/completions"}
    system_prompts = [
        request["json"]["messages"][0]["content"]
        for request in client.requests
    ]
    assert system_prompts == [baseline_prompt] * 10 + [candidate_prompt] * 10
    assert baseline_prompt != candidate_prompt
