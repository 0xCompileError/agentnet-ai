# ruff: noqa: E402, I001
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import agentnet as an
from examples._support import emit


CASES = (
    ("Invoice total is wrong on last month's statement.", "billing"),
    ("Card was charged twice for the same workspace.", "billing"),
    ("Need a copy of the annual invoice.", "billing"),
    ("The dashboard returns a 500 when filters are applied.", "bug"),
    ("Exported CSV files drop the final column.", "bug"),
    ("Mobile sign-in loops back to the login page.", "bug"),
    ("How do I invite a new project member?", "how_to"),
    ("Where can I change notification settings?", "how_to"),
    ("Can I rename an existing workspace?", "how_to"),
    ("What is the retention window for audit logs?", "how_to"),
)
LABELS = frozenset({"billing", "bug", "how_to"})
BASELINE_PROMPT = (
    "Classify each support ticket as billing, bug, or how_to. "
    "Return exactly one label and no other text."
)
CANDIDATE_PROMPT = (
    "Classify each support ticket using only these labels: billing, bug, how_to. "
    "Return exactly one lowercase label and no punctuation."
)
BASELINE_LABELS = (
    "billing",
    "billing",
    "how_to",
    "bug",
    "how_to",
    "bug",
    "how_to",
    "billing",
    "how_to",
    "bug",
)


def score_expected_match(output: object, context: object | None) -> dict[str, object]:
    expected = None
    if isinstance(context, dict):
        expected = context.get("expected_output")
    normalized_output = normalize_label(output)
    score = 1.0 if normalized_output == expected else 0.0
    return {
        "metrics": {
            "matches_expected": score,
            "recognized_label": 1.0 if normalized_output in LABELS else 0.0,
        },
        "passed": score == 1.0,
        "score": score,
    }


def normalize_label(output: object) -> str:
    text = str(output).strip().lower()
    text = text.strip("`\"' .:\n\t")
    for label in LABELS:
        if text == label:
            return label
    return text


def make_dataset() -> an.Dataset:
    return an.Dataset(
        [
            an.TrainingExample(
                input=question,
                expected_output=label,
                id=f"case-{index:02d}",
                metadata={"category": label},
            )
            for index, (question, label) in enumerate(CASES, start=1)
        ],
        name="support-triage-10",
        metadata={"split": "train"},
    )


def make_agent(
    name: str,
    responses: tuple[str, ...],
    *,
    instructions: str,
    llm_mode: str,
    openai_client: object | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> an.ReActAgent:
    llm = make_llm(
        name,
        responses,
        llm_mode=llm_mode,
        openai_client=openai_client,
        api_key=api_key,
        model=model,
    )
    return an.ReActAgent(
        name,
        instructions=instructions,
        llms=[llm],
    )


def make_llm(
    name: str,
    responses: tuple[str, ...],
    *,
    llm_mode: str,
    openai_client: object | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> object:
    if llm_mode == "fake":
        return an.FakeLLM(responses=responses, name=f"{name}-llm")
    if llm_mode == "openai":
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_model = model or os.environ.get("OPENAI_MODEL")
        if not resolved_api_key or not resolved_model:
            raise SystemExit(
                "Set OPENAI_API_KEY and OPENAI_MODEL to run this example with "
                "AGENTNET_TRAINING_LLM=openai"
            )
        return an.OpenAI(
            api_key=resolved_api_key,
            model=resolved_model,
            name=f"{name}-openai",
            client=openai_client,
        )
    raise SystemExit("AGENTNET_TRAINING_LLM must be 'fake' or 'openai'")


def report_openai_progress(event: an.TrainingTrialEvent) -> None:
    """Write concise human progress without disturbing stdout JSON."""

    score = event.validation_score
    if score is None:
        score = event.train_score
    print(
        f"Trial {event.index + 1} {event.net_descriptor['name']} "
        f"{event.status} with score {score:.3f}.",
        file=sys.stderr,
        flush=True,
    )


def run_training(
    *,
    llm_mode: str | None = None,
    openai_client: object | None = None,
    api_key: str | None = None,
    model: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    resolved_llm_mode = (
        llm_mode or os.environ.get("AGENTNET_TRAINING_LLM", "fake")
    ).strip().lower()
    dataset = make_dataset()
    expected_labels = tuple(label for _, label in CASES)

    baseline = make_agent(
        "triage_baseline",
        BASELINE_LABELS,
        instructions=BASELINE_PROMPT,
        llm_mode=resolved_llm_mode,
        openai_client=openai_client,
        api_key=api_key,
        model=model,
    )
    candidate = make_agent(
        "triage_candidate",
        expected_labels,
        instructions=CANDIDATE_PROMPT,
        llm_mode=resolved_llm_mode,
        openai_client=openai_client,
        api_key=api_key,
        model=model,
    )
    trained = an.train(
        baseline,
        dataset,
        optimize=an.ExplicitCandidates([candidate]),
        objective=an.CustomObjective(
            "expected_label_match",
            score_expected_match,
            threshold=1.0,
        ),
        budget=an.Budget(
            max_epochs=1,
            max_examples=20,
            max_llm_calls=20,
            max_trials=2,
        ),
        progress=(
            progress_callback
            if progress_callback is not None
            else report_openai_progress
            if resolved_llm_mode == "openai"
            else None
        ),
    )
    result = trained.training
    completed_trials = [trial for trial in result.trials if trial.status == "completed"]

    def combined_score(trial: an.TrainingTrial) -> float:
        steps = (
            tuple(trial.train_history.steps if trial.train_history is not None else ())
            + tuple(
                trial.validation_history.steps
                if trial.validation_history is not None
                else ()
            )
        )
        return sum(step.score for step in steps) / len(steps)

    return {
        "best_candidate": trained.name,
        "candidates": [
            {
                "name": trial.net.name,
                "passed": bool(
                    trial.train_result is not None
                    and trial.validation_result is not None
                    and trial.train_result.passed
                    and trial.validation_result.passed
                ),
                "prompt": trial.net.instructions,
                "score": combined_score(trial),
            }
            for trial in completed_trials
            if isinstance(trial.net, an.ReActAgent)
        ],
        "checkpoint_count": len(result.trials),
        "dataset_size": len(dataset),
        "evaluated_candidates": len(result.trials),
        "history_steps": len(result.best_trial.train_history.steps)
        + len(result.best_trial.validation_history.steps),
        "is_tied": sum(
            trial.selection_score == result.best_score for trial in completed_trials
        )
        > 1,
        "llm_mode": resolved_llm_mode,
        "passed": bool(
            result.best_trial.train_result is not None
            and result.best_trial.validation_result is not None
            and result.best_trial.train_result.passed
            and result.best_trial.validation_result.passed
        ),
        "score": result.best_score,
        "training_examples": float(
            len(result.best_trial.train_history.steps)
            + len(result.best_trial.validation_history.steps)
        ),
    }


def main() -> None:
    summary = run_training()
    emit("training_10_examples", summary)


if __name__ == "__main__":
    main()
