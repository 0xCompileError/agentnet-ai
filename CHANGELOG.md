# Changelog

All notable changes to AgentNet will be documented in this file.

## [Unreleased]

### Added

- Foundation project scaffold.
- Added `agentnet.train(net, X, y)` and `agentnet.atrain` as the canonical
  end-to-end training APIs with inferred expected-output scoring and validation.
- Added runnable `FittedAgentNet` results, safe `TrainingReport`/`TrainingTrial`
  provenance, bounded `AutoOptimizer`, and `ExplicitCandidates`.
- Added LLM-call limits to training budgets.

### Changed

- Topology optimizers can now be passed directly to `agentnet.train`; users no
  longer need to adapt compiled graphs into executable candidates or write a
  topology scorer callback.
