# GitHub Publishing Runbook

This note records the operational details learned while publishing this repo to
GitHub on 2026-07-30. Use it when moving a local AgentNet directory into its
own GitHub repository or when fixing a failed push.

## Target State

- Repository owner: `0xCompileError`
- Repository name: `agentnet-ai`
- Visibility: public
- Default branch: `main`
- Remote URL: `https://github.com/0xCompileError/agentnet-ai.git`

## Clean Local Repository Setup

Before creating a remote, make sure the local directory is a git repository and
that generated files are ignored.

```bash
git status --short --branch
find . -maxdepth 2 -name .git -type d -print
git log --oneline --decorate -1
```

If there is no `.git` directory, initialize the repo:

```bash
git init -b main
```

Keep generated/runtime artifacts out of the first commit. At minimum, ignore:

```text
.venv/
.pytest_cache/
.ruff_cache/
.coverage
dist/
__pycache__/
```

Then inspect what will be tracked:

```bash
git ls-files --others --exclude-standard
git status --short --ignored
```

## Pre-Push Validation

Run the project validation before committing and pushing:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

For this repo, the full suite passed with `567 passed`.

Do a targeted secret-pattern scan before the first public or private remote
push. Expect intentional hits in tests and docs, but do not push live
credentials.

```bash
rg -n -i "(api[_-]?key|secret|password|token|BEGIN (RSA|OPENSSH|PRIVATE) KEY|sk-[A-Za-z0-9])" \
  --glob "!dist/**" \
  --glob "!.venv/**" \
  --glob "!.pytest_cache/**" \
  --glob "!.ruff_cache/**" \
  --glob "!**/__pycache__/**" \
  .
```

## GitHub CLI Authentication

`gh auth status` can report an invalid saved token even when an account name is
present. Check it explicitly:

```bash
gh auth status
```

If the token is invalid, use the browser device flow. The interactive prompt can
stall in non-interactive PTYs, so pipe the answer to the git-credential question:

```bash
printf "Y\n" | gh auth login --hostname github.com --git-protocol https --web --clipboard
open -a "Google Chrome" https://github.com/login/device
```

Complete the browser approval with the one-time code printed by `gh`.

## Workflow Scope Gotcha

If the commit includes `.github/workflows/*.yml`, GitHub may reject the push
unless the token has the `workflow` scope:

```text
refusing to allow an OAuth App to create or update workflow `.github/workflows/ci.yml` without `workflow` scope
```

Refresh the token with the extra scope:

```bash
printf "Y\n" | gh auth refresh --hostname github.com --scopes workflow --clipboard
open -a "Google Chrome" https://github.com/login/device
```

Approve the scope refresh in the browser, then retry the push.

## Create And Push Private Repo

First check whether the repository already exists:

```bash
gh repo view 0xCompileError/agentnet --json nameWithOwner,visibility,url
```

If it does not exist, create it private and push the current branch:

```bash
gh repo create 0xCompileError/agentnet \
  --private \
  --source . \
  --remote origin \
  --push \
  --description "PyTorch-inspired framework for trainable networks of ReAct agents"
```

If creation succeeds but push fails, the remote may already be configured. Retry
with plain git after fixing auth scopes:

```bash
git push -u origin main
```

## Verify Remote State

Verify both local tracking and the remote branch SHA:

```bash
git status --short --branch
git remote -v
git ls-remote origin refs/heads/main
```

For the initial AgentNet push, `main` was pushed to:

```text
15515a99bb7d82d7c05e47537931276928c2add2 refs/heads/main
```

## Add A Teammate

Use the GitHub UI when CLI auth is questionable:

```text
Repository -> Settings -> Collaborators and teams -> Add people
```

If `gh auth status` is healthy and the token has repository administration
permissions, add a collaborator from the CLI:

```bash
gh repo add-collaborator 0xCompileError/agentnet TEAMMATE_USERNAME --permission push
```

Use `--permission maintain` only when the teammate should manage settings and
branches, not just push code.
