"""Training primitives for eval-driven AgentNet optimization."""

from agentnet.training.attribution import AttributionEngine, AttributionRecord
from agentnet.training.automatic import (
    AutoOptimizer,
    ExplicitCandidates,
    TrainingCandidate,
)
from agentnet.training.budget import Budget, BudgetManager
from agentnet.training.checkpoints import TrainingCheckpoint
from agentnet.training.datasets import Dataset, TrainingExample
from agentnet.training.history import TrainingHistory, TrainingStep
from agentnet.training.optimizers import (
    FallbackOptimizationResult,
    FallbackOptimizer,
    RetryPolicyOptimizationResult,
    RetryPolicyOptimizer,
)
from agentnet.training.patches import TrainingPatch, generate_training_patch
from agentnet.training.progress import TrainingProgressCallback, TrainingProgressEvent
from agentnet.training.simple import (
    FittedAgentNet,
    TrainingReport,
    TrainingTrial,
    TrainingTrialEvent,
    atrain,
    train,
)
from agentnet.training.trainer import Trainer, TrainingCandidateResult, TrainingResult

__all__ = [
    "AttributionEngine",
    "AttributionRecord",
    "AutoOptimizer",
    "Budget",
    "BudgetManager",
    "Dataset",
    "ExplicitCandidates",
    "FallbackOptimizationResult",
    "FallbackOptimizer",
    "RetryPolicyOptimizationResult",
    "RetryPolicyOptimizer",
    "Trainer",
    "FittedAgentNet",
    "TrainingCandidate",
    "TrainingCandidateResult",
    "TrainingCheckpoint",
    "TrainingExample",
    "TrainingHistory",
    "TrainingPatch",
    "TrainingProgressCallback",
    "TrainingProgressEvent",
    "TrainingReport",
    "TrainingResult",
    "TrainingStep",
    "TrainingTrial",
    "TrainingTrialEvent",
    "atrain",
    "generate_training_patch",
    "train",
]
