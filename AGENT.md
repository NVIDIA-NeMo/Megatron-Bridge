# Agent Guide

This document provides structured instructions for AI agents operating in this repository.

---

## 1. CI/CD Overview

### Container Build

The CI container is built from `docker/Dockerfile.ci` and pushed to the Azure Container Registry `nemoci.azurecr.io`. To build and run it locally:

```bash
# Build
docker build -f docker/Dockerfile.ci -t megatron-bridge .

# Run interactive shell
docker run --rm -it -w /workdir -v $(pwd):/opt/Megatron-Bridge \
  --entrypoint bash --gpus all megatron-bridge
```

Key build args:
- `FROM_IMAGE_NAME` — base PyTorch image (default: `nvcr.io/nvidia/pytorch:25.09-py3`)
- `MCORE_TRIGGERED_TESTING` — set to `true` when testing against a non-pinned MCore commit
- `MCORE_COMMIT_SHA` — specific MCore commit SHA to pin during build

### Dependency Management

Dependencies are managed via `uv` with `pyproject.toml` and `uv.lock`.

**Never install or upgrade dependencies outside the CI container.** All `uv` commands must be run inside a `megatron-bridge` container — either one you built locally or a pre-built image.

```bash
# Build the container (if no pre-built image is available)
docker build -f docker/Dockerfile.ci -t megatron-bridge .

# Install dependencies inside the container
docker run --rm -v $(pwd):/workdir/ -w /workdir/ megatron-bridge \
  uv pip install -e ".[dev]"

# Update the lock file inside the container
docker run --rm -v $(pwd):/workdir/ -w /workdir/ megatron-bridge uv lock
```

Rules:

- Never run `uv pip install`, `uv lock`, or any dependency command on the host — always use the container.
- Use `uv run --no-sync --active` for local tool invocations inside the container.
- The lock file must be kept up to date; CI will fail if it is stale.
- Custom package sources include the NVIDIA PyPI index — do not remove these.

### Linting and Formatting

Before committing, run:

```bash
ruff check --fix <changed_files>
ruff format <changed_files>
pre-commit run --all-files
```

### Test Onboarding

Tests live under `tests/`:

| Path | Description |
|------|-------------|
| `tests/unit_tests/` | Fast, isolated unit tests grouped by domain (models, core, data, etc.) |
| `tests/functional_tests/` | Integration tests with models/datasets, tiered L0/L1/L2 |

**Adding a unit test:**
1. Place it under `tests/unit_tests/<domain>/test_<name>.py`.
2. Use the appropriate pytest marker: `@pytest.mark.unit`.
3. Run locally: `uv run --no-sync --active pytest tests/unit_tests/<your_test>.py`
4. Or inside Docker: `docker run --rm --gpus all -v $(pwd):/workdir/ -w /workdir/ megatron-bridge uv run pytest tests/unit_tests/`

**Adding a functional test:**
1. Create a launch script under `tests/functional_tests/launch_scripts/active/`.
2. Follow the naming convention: `L0_Launch_<area>_<desc>.sh`, `L1_Launch_...`, or `L2_Launch_...`.
3. Tier guidance:
   - **L0** — smoke tests that run on every PR; must be fast and stable.
   - **L1** — broader coverage; runs nightly.
   - **L2** — heavy tests (large models, checkpoint conversion); runs on schedule or manual trigger.
4. Apply the `needs-more-tests` PR label to trigger L0 + L1 for a PR.

**Pytest markers available:** `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`

---

## 2. CI Failure Assistance

### Locating the PR from a CI Branch

CI runs are triggered by pushes to branches following the pattern `pull-request/<number>`. To find the associated PR:

```bash
# Extract PR number from the CI branch name (e.g. pull-request/1234)
PR_NUMBER=$(git rev-parse --abbrev-ref HEAD | grep -oP '(?<=pull-request/)\d+')

# Or, given a branch name string directly:
PR_NUMBER=$(echo "pull-request/1234" | grep -oP '(?<=pull-request/)\d+')

# Fetch PR metadata
gh pr view "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge

# List files changed in the PR
gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge --name-only

# View PR checks / CI status
gh pr checks "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
```

### Investigating a Failing CI Job

1. **Get the PR number** from the branch name (see above).
2. **Review the changeset** to understand what changed:
   ```bash
   gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
   ```
3. **Identify the failing job** from `gh pr checks` output or from the GitHub Actions URL in the failure notification.
4. **Fetch job logs** for deeper inspection:
   ```bash
   # List runs for the PR's head SHA
   gh run list --repo NVIDIA-NeMo/Megatron-Bridge --branch "pull-request/$PR_NUMBER"

   # View logs for a specific run
   gh run view <run_id> --repo NVIDIA-NeMo/Megatron-Bridge --log-failed
   ```
5. **Cross-reference the changeset** against the failing test or step to narrow down the root cause.

### Common Failure Patterns

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Lint job fails | `ruff` or `pre-commit` violation | Run `ruff check --fix` + `ruff format` locally |
| Container build fails | Dependency conflict or stale `uv.lock` | Re-run `uv lock` inside Docker and commit updated lock |
| Unit tests fail | Code regression or missing import | Run failing test locally; check the PR diff for the relevant module |
| Functional test (L0) fails | Integration breakage | Check GPU runner logs; reproduce with the corresponding `L0_Launch_*.sh` script |
| `cicd-wait-in-queue` blocked | PR not yet approved for CI | A maintainer must comment `/ok to test <SHA>` or approve via the test queue |
| MCore submodule mismatch | Pinned commit out of sync | Update `3rdparty/Megatron-LM` submodule and re-lock |

### CI Pipeline Structure

```
pre-flight
  └── lint-check
        └── cicd-wait-in-queue          # requires maintainer approval for untrusted PRs
              └── cicd-container-build  # builds and caches the Docker image
                    ├── unit-tests-core
                    ├── unit-tests-diffusion
                    └── functional-tests (L0 always; L1 with needs-more-tests label; L2 on schedule)
```

- The CI branch `pull-request/<number>` is created automatically when a PR is opened against `main` or `deploy-release/*`.
- Concurrent runs for the same PR are cancelled automatically (concurrency group per PR number).
- Slack notifications are sent on completion for scheduled and nightly runs.
