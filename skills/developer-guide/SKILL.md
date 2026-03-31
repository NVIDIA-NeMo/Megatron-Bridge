---
name: developer-guide
description: Developer environment setup and best practices for Megatron Bridge. Covers container-based development, uv package management, pre-commit hooks, running tests, and common pitfalls. Use when onboarding, setting up a dev environment, or troubleshooting build issues.
---

# Developer Guide

This guide covers the recommended development workflow for Megatron Bridge.
Two core principles apply everywhere: **build and develop inside containers**,
and **always use uv** for package management.

---

## Why Containers

Megatron Bridge depends on CUDA, NCCL, PyTorch with GPU support,
Transformer Engine, and optional components like TRT-LLM, vLLM, and DeepEP.
Installing these on a bare host is fragile and hard to reproduce. The project
ships production-quality Dockerfiles that pin every dependency.

**Use the container as your development environment.** This guarantees:

- Identical CUDA / NCCL / cuDNN versions across all developers and CI.
- `uv.lock` resolves the same way locally and in CI (the lockfile is
  Linux-only; it cannot be regenerated on macOS).
- GPU-dependent operations (training, conversion, `uv lock`) work out of the
  box.

### Option 1: Use the NeMo Framework Container

The fastest way to get started is the pre-built
[NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags),
which ships with Megatron Bridge, Megatron-Core, and all GPU dependencies
pre-installed. No build step required:

```bash
docker run --rm -it --gpus all --shm-size=24g \
  nvcr.io/nvidia/nemo:latest \
  bash
```

### Option 2: Build the Megatron Bridge Container

If you need to test against your local source tree, build the image from the
repository root:

```bash
docker build \
  -f docker/Dockerfile.ci \
  --target megatron_bridge \
  -t megatron-bridge:latest \
  .
```

This builds the CI image with all dependencies installed via `uv sync --locked`.
See `docker/README.md` for the full NeMo Framework image stack
(fw-base -> megatron-bridge -> fw-final) and build argument reference.

### Running the Container

Interactive development shell:

```bash
docker run --rm -it -w /opt/Megatron-Bridge \
  -v $(pwd):/opt/Megatron-Bridge \
  --gpus all \
  --shm-size=24g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  megatron-bridge:latest \
  bash
```

### Containers on Slurm Clusters

On Slurm clusters with Enroot/Pyxis, containers are passed to `srun` directly:

```bash
srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd /opt/Megatron-Bridge && uv run --no-sync python ..."
```

If you use the built container (or the NeMo Framework container) as-is,
dependencies are already installed and no `uv sync` is needed. If you
**bind-mount a custom Megatron Bridge source tree** into the container
(e.g., for development), you need to `uv sync` so dependencies match
your local `pyproject.toml` and `uv.lock`. In that case, only rank 0
should sync while other ranks wait:

```bash
if [ "$SLURM_LOCALID" -eq 0 ]; then uv sync; else sleep 10; fi
```

Other key points:

- `--no-container-mount-home` is an **srun flag**, not an `#SBATCH` directive.
- Set `UV_CACHE_DIR` to shared storage to avoid filling the container's
  `/root/.cache/`.

---

## Always Use uv

Megatron Bridge uses [uv](https://docs.astral.sh/uv/) as its sole package
manager. The `uv.lock` file is checked into the repository for reproducible
builds. **Never use `pip install`, `conda`, or bare `python`** — always go
through `uv`.

### Why uv

- **Reproducibility**: `uv.lock` pins every transitive dependency, ensuring
  identical environments across developers, CI, and production containers.
- **Speed**: uv resolves and installs dependencies 10-100x faster than pip.
- **Single tool**: uv handles virtual environments, dependency resolution,
  locking, syncing, and running scripts — no need for separate tools.
- **CI integration**: `Dockerfile.ci` installs everything via
  `uv sync --locked`. If you use pip to install something locally, it will
  diverge from what CI tests against.
- **Cache-friendly**: Set `UV_CACHE_DIR` to a persistent host directory and
  mount it into the container to avoid re-downloading wheels on every
  `docker run`. This is especially useful when you mount a frequently
  changing workdir that triggers re-syncs:
  ```bash
  docker run --rm -it \
    -v $(pwd):/opt/Megatron-Bridge \
    -v $HOME/.cache/uv:/root/.cache/uv \
    --gpus all --shm-size=24g \
    megatron-bridge:latest bash
  ```

### Essential uv Commands

| Task | Command |
|---|---|
| Install all deps from lockfile | `uv sync --locked` |
| Install with all extras and dev groups | `uv sync --locked --all-extras --all-groups` |
| Run a Python command | `uv run python script.py` |
| Run training | `uv run python -m torch.distributed.run --nproc_per_node=N script.py` |
| Add a new dependency | `uv add <package>` |
| Add an optional dependency | `uv add --optional --extra <group> <package>` |
| Regenerate the lockfile | `uv lock` (must be done inside the container on Linux) |
| Run linting | `uv run ruff check --fix . && uv run ruff format .` |
| Install pre-commit hooks | `uv run --group dev pre-commit install` |

### uv run, Not bare python

Always launch scripts with `uv run`:

```bash
# Correct
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py ...

# Wrong — bypasses the uv-managed environment
python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py ...
torchrun --nproc_per_node=1 scripts/training/run_recipe.py ...
```

After running `uv sync` inside a container, you can also use bare `python`
since the virtual environment is already activated. But `uv run` is always the
safer default.

### Adding Dependencies

```bash
uv add some-package

# For an optional extra group (e.g., trtllm-specific deps)
uv add --optional --extra trtllm some-package
```

This updates `pyproject.toml` and `uv.lock`. Commit both files:

```bash
git add pyproject.toml uv.lock
git commit -s -m "build: add some-package dependency"
```

### Regenerating uv.lock

The lockfile is Linux-only (it resolves against CUDA wheels). **You cannot
regenerate it on macOS.** Run `uv lock` inside the Docker container or on a
Linux workstation:

```bash
docker run --gpus all --rm \
  -v $(pwd):/opt/Megatron-Bridge \
  megatron-bridge:latest \
  bash -c 'cd /opt/Megatron-Bridge && uv lock'
```

### uv sync After Switching MCore Branches

The lockfile is generated against the main MCore commit. When switching to the
dev branch:

```bash
./scripts/switch_mcore.sh dev
uv sync            # without --locked
```

When switching back to main:

```bash
./scripts/switch_mcore.sh main
uv sync --locked   # lockfile matches again
```

---

## Pre-commit Hooks

Install pre-commit hooks before your first commit:

```bash
uv run --group dev pre-commit install
```

The hooks run [ruff](https://docs.astral.sh/ruff/) for linting and formatting,
plus end-of-file and trailing-whitespace fixers. If hooks auto-fix files,
re-stage and re-run:

```bash
git add -u
pre-commit run
# If it auto-fixed files:
git add -u
pre-commit run
```

Repeat until all hooks pass.

---

## Running Tests

The recommended way to run the full test suite locally is
`scripts/run_ci_tests.sh`, which mirrors the GitHub CI pipeline (lint, unit
tests, functional tests, coverage):

```bash
bash scripts/run_ci_tests.sh                    # local mode, GPUs 0,1
bash scripts/run_ci_tests.sh --mode docker       # build container and run inside it
bash scripts/run_ci_tests.sh --gpus 0 --skip-functional  # unit + lint only on GPU 0
bash scripts/run_ci_tests.sh --skip-lint --skip-unit      # functional tests only
```

You can also run individual test suites directly:

### Unit Tests

```bash
uv run pytest tests/unit_tests/ -x -v
```

Unit tests run without GPUs and do not depend on large artifacts.

### Functional Tests

Functional tests require GPUs and are typically run inside the container:

```bash
uv run pytest tests/functional_tests/ -x -v
```

Longer functional tests use `L2_Launch_*.sh` launcher scripts in
`tests/functional_tests/`. Each launcher must be registered in
`.github/workflows/cicd-main.yml` under `matrix.include` to be picked up
by CI.

---

## Commit and PR Workflow

- **Never commit directly to `main`** — always create a feature branch.
- **Always sign commits**: `git commit -s -m "message"`.
- **PR title format**: `[{areas}] {type}: {description}`
  (e.g., `[model] feat: Add Qwen3 model bridge`).
- **Trigger CI**: Comment `/ok to test <commit-sha>` on the PR, or set up
  signed commits for automatic CI triggering.

See `CONTRIBUTING.md` for the full PR workflow, area/type labels, and DCO
requirements.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `uv sync --locked` fails on macOS | Lockfile resolves CUDA wheels that don't exist on macOS | Run inside Docker or on a Linux machine |
| `ModuleNotFoundError` after pip install | pip installed outside the uv-managed venv | Use `uv add` and `uv sync`, never bare `pip install` |
| `uv sync --locked` fails after MCore branch switch | Lockfile was generated against main MCore | Use `uv sync` (without `--locked`) on dev |
| Stale checkpoint auto-resume in Bridge | `nemo_experiments/` from a previous run exists | `rm -rf nemo_experiments` before starting fresh |
| Port collision on Slurm (EADDRINUSE) | `ntasks-per-node=8` with `torchrun --nproc_per_node=8` | Drop torchrun; use `ntasks-per-node=8` with `uv run python script.py` (srun-native) |
| `uv: command not found` inside container | Container doesn't have uv | Use the `megatron-bridge` image built from `Dockerfile.ci` |
| `No space left on device` during uv ops | Cache fills container's `/root/.cache/` | Set `UV_CACHE_DIR` to shared/persistent storage |
| Pre-commit fails with ruff errors | Code style violations | Run `uv run ruff check --fix . && uv run ruff format .` |

---

## Quick Start Checklist

1. Clone the repo and initialize submodules:
   ```bash
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge megatron-bridge
   cd megatron-bridge
   git submodule update --init 3rdparty/Megatron-LM
   ```

2. Build the container:
   ```bash
   docker build -f docker/Dockerfile.ci --target megatron_bridge -t megatron-bridge:latest .
   ```

3. Start a dev shell:
   ```bash
   docker run --rm -it -v $(pwd):/opt/Megatron-Bridge --gpus all --shm-size=24g megatron-bridge:latest bash
   ```

4. Install pre-commit hooks (inside container):
   ```bash
   uv run --group dev pre-commit install
   ```

5. Run a quick training sanity check:
   ```bash
   uv run python -m torch.distributed.run --nproc_per_node=1 \
     scripts/training/run_recipe.py \
     --recipe vanilla_gpt_pretrain_config \
     train.train_iters=5 train.global_batch_size=8 train.micro_batch_size=4 \
     scheduler.lr_warmup_iters=1 scheduler.lr_decay_iters=5 \
     logger.log_interval=1
   ```

6. Create a branch, make changes, and submit a PR:
   ```bash
   git switch -c your-feature-name
   # ... make changes ...
   git add -u && git commit -s -m "[area] type: description"
   git push origin your-feature-name
   ```
