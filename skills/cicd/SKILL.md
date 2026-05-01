---
name: cicd
description: CI/CD reference for Megatron Bridge — pipeline structure, commit and PR workflow, CI failure investigation, and common failure patterns.
when_to_use: Investigating a CI failure, understanding the pipeline structure, writing a commit or PR, triggering CI, 'CI is red', 'how do I trigger CI', 'PR workflow', 'where are the logs', 'cicd-wait-in-queue blocked'.
---

# CI/CD

## Commit and PR Workflow

- **Never commit directly to `main`** — always create a feature branch.
- **Always sign commits**: `git commit -s -m "message"`.
- **PR title format**: `[{areas}] {type}: {description}`
  (e.g., `[model] feat: Add Qwen3 model bridge`).
- **Trigger CI**: comment `/ok to test <commit-sha>` on the PR, or set up
  signed commits for automatic triggering.

See @CONTRIBUTING.md for the full PR workflow, area/type labels, and DCO requirements.

## Pipeline Structure

Defined in `.github/workflows/cicd-main.yml`. Triggered by schedule, pushes
to `main` / `deploy-release/*`, merge groups, and `workflow_dispatch`.

```
pre-flight
  └── lint-check
        └── cicd-wait-in-queue       # requires maintainer approval for untrusted PRs
              └── cicd-container-build
                    ├── unit-tests-core
                    ├── unit-tests-diffusion
                    └── functional-tests (L0 always; L1 with needs-more-tests label; L2 on schedule)
```

- CI branch `pull-request/<number>` is created automatically when a PR is opened against `main` or `deploy-release/*`.
- Concurrent runs for the same PR are cancelled automatically (concurrency group per PR number).
- Slack notifications are sent on completion for scheduled and nightly runs.

For functional test tier semantics and job-to-directory mapping, see the `testing` skill.

## CI Failure Investigation

### Locating the PR from a CI Branch

```bash
# Extract PR number from branch name (e.g. pull-request/1234)
PR_NUMBER=$(git rev-parse --abbrev-ref HEAD | grep -oP '(?<=pull-request/)\d+')

gh pr view "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge --name-only
gh pr checks "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
```

### Investigating a Failing Job

1. **Get the PR number** from the branch name (see above).
2. **Review the changeset**:
   ```bash
   gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
   ```
3. **Identify the failing job** from `gh pr checks` output.
4. **Fetch job logs**:
   ```bash
   gh run list --repo NVIDIA-NeMo/Megatron-Bridge --branch "pull-request/$PR_NUMBER"
   gh run view <run_id> --repo NVIDIA-NeMo/Megatron-Bridge --log-failed > run.log
   ```
5. **Scan logs in chunks** — log files can exceed 10,000 lines, never load them whole:
   ```bash
   wc -l run.log
   tail -200 run.log          # start from the end
   sed -n '1,200p' run.log    # or scan forward in 200-line chunks
   ```
6. **Cross-reference the changeset** against the failing step.

## Common Failure Patterns

| Symptom | Likely Cause | Action |
|---|---|---|
| Lint job fails | `ruff` or `pre-commit` violation | Run `ruff check --fix` + `ruff format` locally |
| Container build fails | Dependency conflict or stale `uv.lock` | Re-run `uv lock` inside Docker and commit updated lock |
| Unit tests fail | Code regression or missing import | Run failing test locally; check the PR diff |
| Functional test (L0) fails | Integration breakage | Check GPU runner logs; reproduce with `L0_Launch_*.sh` |
| `cicd-wait-in-queue` blocked | PR not yet approved for CI | Maintainer must comment `/ok to test <SHA>` |
| MCore submodule mismatch | Pinned commit out of sync | Update `3rdparty/Megatron-LM` submodule and re-lock |
| Stale checkpoint auto-resume | `nemo_experiments/` from a previous run exists | `rm -rf nemo_experiments` before starting fresh |
| Port collision on Slurm (EADDRINUSE) | `ntasks-per-node=8` with `torchrun` | Drop torchrun; use `ntasks-per-node=8` with `uv run python script.py` |
