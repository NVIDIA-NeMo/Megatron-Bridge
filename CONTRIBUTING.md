# Contributing To Megatron-Bridge

Thanks for your interest in contributing to Megatron-Bridge!

## 🛠️ Setting Up Your Environment

### Development Environment

You can either follow the steps below to set up the environment from scratch, or use the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags), which provides a pre-built environment and makes these steps unnecessary.

**Build and run the Docker container**:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t megatron-bridge \
    .
```

To start a shell in the container to interactively run/develop:

```bash
docker run --rm -it -w /workdir -v $(pwd):/opt/Megatron-Bridge \
  --entrypoint bash \
  --gpus all \
  megatron-bridge
```

If you are using VSCode/Cursor you can also use Dev Containers. Here's a devcontainer.json to get you started:

```jsonc
{
    "name": "megatron-bridge-dev",
    "image": "megatron-bridge:latest",
    "runArgs": [
        "--gpus",
        "all",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--shm-size=24g",
        "--privileged",
        "--pid=host"
    ]

    // NOTE: Here is an example of how you can set up some common mounts, environment variables, and set up your shell.
    //       Feel free to adapt to your development workflow and remember to replace the paths with your username.

    //"mounts": [
    //    {"source": "/home/yourusername", "target": "/home/yourusername", "type": "bind"},
    //    {"source": "/home/yourusername/.ssh", "target": "/root/yourusername-ssh", "type": "bind"}
    //],
    //"containerEnv": {
    //    "HF_TOKEN_PATH": "/home/yourusername/.cache/huggingface/token",
    //    "HF_HOME": "/home/yourusername/.cache/huggingface",
    //    "HF_DATASETS_CACHE": "/home/yourusername/.cache/huggingface/datasets",
    //    "WANDB_API_KEY": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    //},
    // // This (1) marks all directories safe (2) copies in ssh keys (3) sources user's bashrc file
    //"postStartCommand": "git config --global --add safe.directory '*' && cp -r /root/yourusername-ssh/* /root/.ssh/ && source /home/yourusername/.bashrc"
}
```

## 🔄 Making Changes

### Workflow: For External Contributors (Fork Required)

If you're an external contributor, you'll need to fork the repository:

1. **Create a fork**: Click the "Fork" button on the [GitHub repository page](https://github.com/NVIDIA-NeMo/Megatron-Bridge) or follow this [direct link to fork](https://github.com/NVIDIA-NeMo/Megatron-Bridge/fork)

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Megatron-Bridge megatron-bridge
   cd megatron-bridge
   ```

3. **Add upstream remote** to keep your fork updated:
   ```bash
   git remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
   ```

4. **Install pre-commit**:
   ```bash
   # Requires `uv` to be installed
   uv run --group dev pre-commit install
   ```

5. **Keep your fork updated** before starting new work:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

6. **Create a new branch** for your changes:
   ```bash
   git checkout main
   git switch -c your-feature-name
   ```

7. **Make your changes and commit** them:
   ```bash
   git add .
   git commit --signoff -m "Your descriptive commit message"
   ```

   We require signing commits with `--signoff` (or `-s` for short). See [Signing Your Work](#signing-your-work) for details.

8. **Push to your fork**:
   ```bash
   git push origin your-feature-name
   ```

9. **Create a pull request** from your fork's branch to the main repository's `main` branch through the GitHub web interface.

### Workflow: For NVIDIA Contributors (Direct Access)

If you have write access to the repository (NVIDIA contributors):

1. **Clone the repository** directly:
   ```bash
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge megatron-bridge
   cd megatron-bridge
   ```

2. **Install pre-commit** from the project root directory:
   ```bash
   # Requires `uv` to be installed
   uv run --group dev pre-commit install
   ```

3. **Create a new branch** for your changes:
   ```bash
   git switch -c your-feature-name
   ```

4. **Make your changes and commit** them:
   ```bash
   git add .
   git commit --signoff -m "Your descriptive commit message"
   ```

5. **Push your branch** to the repository:
   ```bash
   git push origin your-feature-name
   ```

6. **Create a pull request** from your branch to the `main` branch.

## 📋 Commit and PR Title Format

Format your commit messages and PR titles as:

```text
[{areas}] {type}: {description}
```

**Areas** (use the most relevant ones, separate multiple with `,`):
- `model` - Model implementations and HF bridge logic
- `recipe` - Training recipes and launch configs
- `training` - Training loop, callbacks, and runtime integration
- `data` - Dataset builders, preprocessing, and samplers
- `ckpt` - Checkpoint conversion, loading, export, and save paths
- `peft` - PEFT methods (LoRA, adapters) and adapter export
- `perf` - Performance optimizations and throughput improvements
- `distill` - Knowledge distillation
- `prune` - Pruning and sparsity
- `quant` - Quantization (PTQ, QAT, FP8 recipes)
- `diffusion` - Diffusion model implementations and training
- `ci` - CI, automation, and workflow infrastructure
- `docs` - Documentation, examples, and contributor guidance
- `build` - Dependencies, packaging, and environment setup
- `misc` - Cross-cutting utilities and other changes

**Types**:
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code refactoring without changing functionality
- `chore` - Maintenance tasks
- `test` - Adding or updating tests

**Breaking Changes**: If your PR breaks any API (CLI arguments, config, function signature, etc.), add `[BREAKING]` to the beginning of the title.

**Examples**:
```text
[model] feat: Add Qwen3 model bridge
[recipe, docs] feat: Add Llama 3.1 70B recipe with documentation
[ckpt] fix: Handle missing keys in HF checkpoint conversion
[BREAKING][training] refactor: Change optimizer config structure
[ci, build] chore: Update ruff version
```

## 🏷️ Labeling Your PR

When you create a pull request, **add labels immediately** so reviewers and CI can route it correctly. At minimum, apply:

1. **One type label** — `bug`, `feature`, `docs`, or `ci`
2. **One or more area labels** — `area:model`, `area:recipe`, `area:training`, `area:data`, `area:ckpt`, `area:peft`, `area:perf`, `area:distill`, `area:diffusion`, `area:prune`, `area:quant`, `area:build`, or `area:misc`
3. **`docs-only`** — if the PR touches only documentation (no code changes); this skips most CI jobs
4. **`needs-review`** — when the PR is ready for review
5. **`needs-more-tests`** — if the change needs additional test coverage; triggers both L0 and L1 CI
6. **`high-complexity`** — if the PR is large, touches many files, or is prone to merge conflicts

Add risk labels when applicable:
- `breaking-change` — if any public API, CLI argument, config key, or function signature changes
- `needs-more-tests` — if the change needs additional test coverage (also triggers L1 CI)

Properly labeled PRs get faster reviews and avoid sitting in the triage queue.

## 🏷️ Repository Labels and Triage

Megatron Bridge uses a small governance taxonomy so maintainers, oncall, and automation can reason about issues and PRs consistently:

- New issues should start with `needs-triage` and leave triage with one `type` label plus one `area` label.
- PRs should use one primary `area:*` value in the PR template. State labels such as `needs-author`, `blocked`, and `ready-to-merge` are for routing active work, not for replacing review status or CI details.
- Release labels such as `r0.3.0`, community labels, and `needs-follow-up` are still valid, but they are orthogonal to the main governance taxonomy.

### Type Labels

Use exactly one type label per issue or PR after triage:

| Label | Use for |
| --- | --- |
| `bug` | Incorrect behavior, regressions, or broken workflows |
| `feature` | New capabilities, enhancements, or enablement work |
| `support` | Questions, help requests, or user guidance gaps |
| `docs` | Documentation-only updates or documentation debt |
| `ci` | CI, automation, test queue, or workflow infrastructure work |

### State Labels

Use at most one primary state label from this set at a time (see exceptions below):

| Label | Meaning |
| --- | --- |
| `needs-triage` | New item needs classification and ownership |
| `needs-review` | PR is ready for code review and waiting on a reviewer |
| `needs-author` | Author action is required before review or merge can continue |
| `needs-follow-up` | Issue or PR has finished initial triage/review and needs further follow-up |
| `blocked` | Work cannot move forward until an external dependency is cleared |
| `ready-to-merge` | PR is approved, current, and only waiting for CI to pass before merge |

**Allowed combinations:** `needs-author` + `needs-follow-up` and `needs-follow-up` + `blocked` can co-exist (e.g., waiting on the author but oncall should keep tracking, or a blocked item that oncall should keep watching across handoffs).

### Risk Labels

Apply only when risk affects review or merge behavior:

| Label | Meaning |
| --- | --- |
| `breaking-change` | Public behavior or API compatibility changes |
| `high-complexity` | Harder to merge: prone to conflicts and needs additional test coverage |
| `needs-more-tests` | Requires additional test coverage; triggers both L0 and L1 CI test tiers |

### Area Labels

Use one primary area label after triage:

| Label | Scope |
| --- | --- |
| `area:model` | Model implementations and HF bridge logic |
| `area:recipe` | Training recipes and launch configs |
| `area:training` | Training loop, callbacks, and runtime integration |
| `area:data` | Dataset builders, preprocessing, and samplers |
| `area:ckpt` | Checkpoint conversion, loading, export, and save paths |
| `area:peft` | PEFT methods (LoRA, adapters) and adapter export |
| `area:perf` | Performance optimizations, kernel integration, and throughput improvements |
| `area:distill` | Knowledge distillation |
| `area:diffusion` | Diffusion model implementations and training |
| `area:prune` | Pruning and sparsity |
| `area:quant` | Quantization (PTQ, QAT, FP8 recipes) |
| `area:build` | Dependencies, packaging, images, and environment setup |
| `area:misc` | Cross-cutting utilities, logging, helpers, and other changes that do not fit a primary domain |

### Orthogonal Labels

This taxonomy does not replace every existing label:

- Keep release labels such as `r0.3.0` as independent scheduling signals.
- Keep `community-request` and other community-related labels as independent intake signals.
- Use `needs-follow-up` when an issue or PR should stay explicitly visible to the oncaller across handoffs.
- Avoid creating new status synonyms when an existing label in this taxonomy already fits.

### Label Application Rules

- New issues should start with `needs-triage`.
- Issues should leave triage with one `type` label and one `area` label.
- An issue keeps `needs-triage` until a maintainer has responded or assigned it. Adding type and area labels is classification; the issue leaves `needs-triage` only when a maintainer engages (responds, assigns, or explicitly routes it).
- After a maintainer engages, transition to `needs-follow-up` (deferred work oncall should track), `needs-author` (waiting on reporter for more info), `blocked` (external dependency), or no state label (actively being worked on).
- PRs should not use `needs-triage`. Use `needs-review`, `needs-author`, `blocked`, or `ready-to-merge` only when they help route work.
- `high-complexity` starts as a manual maintainer label, not an automated heuristic.
- `needs-follow-up` should usually point to a linked issue instead of staying on a merged PR.
- `needs-follow-up` is the visibility label for deferred work that should stay on the oncall radar.
- `needs-follow-up` can be combined with `blocked` when the oncaller should keep watching a blocked item.
- If a PR is marked `breaking-change`, do not treat it as auto-mergeable even if CI is green.

### Daily Views

These four views are the core daily queues maintainers and oncall should watch.

#### Needs Triage

- Scope: open issues labeled `needs-triage`
- Goal: assign one `type` and one `area`
- Suggested query: `is:issue is:open label:"needs-triage" sort:updated-asc`

#### Ready To Merge

- Scope: open PRs labeled `ready-to-merge`
- Goal: surface PRs that should merge without rereading every CI detail
- Suggested query: `is:pr is:open label:"ready-to-merge" draft:false sort:updated-asc`

#### Blocked Or Needs Follow-Up

- Scope: open issues and PRs labeled `blocked` or `needs-follow-up`
- Goal: make blockers and deferred work visible across handoffs
- Suggested query: `is:open (label:"blocked" OR label:"needs-follow-up") sort:updated-asc`

#### High Complexity

- Scope: open PRs labeled `high-complexity`
- Goal: proactively review, rebase, and ensure adequate test coverage before conflicts waste CI and reviewer time
- Suggested query: `is:pr is:open label:"high-complexity" sort:updated-asc`

#### Recommended Columns

If you mirror these queues into a GitHub Project, keep the columns and sort keys small:

- item title
- primary area
- owner or assignee
- age
- last updated time
- release label
- current state

## 📝 Writing Tests

We use [pytest](https://docs.pytest.org/en/stable/) for writing both unit and functional tests.

**Unit tests** aim to test functions in isolation. They generally do not depend on artifacts like Hugging Face checkpoints or larger datasets. Exception to this is a small toy dataset consisting of tokenizers.
Unit tests are stored at `tests/unit_tests`. Please add your test to an existing folder or create a new one if none matches.

> **Preferred:** Unit tests are strongly recommended over functional tests. CI resources are limited, so every functional test slot has a real cost. Cover as much logic as possible with unit tests before reaching for a functional test.

**Functional tests** are integration tests that perform model training or operate on larger artifacts. We use pytest for writing these. In some cases, it might be desired to run your test (or parts of it) in a subprocess to avoid process contamination. We use `subprocess.run` for this inside the pytest function. Please add your test into one of the predefined folders. If none of the folders matches semantically, please reach out to the `@nvidia-nemo/automation` in your PR for consultation.

> **GPU limit:** Functional tests must use **at most 2 GPUs**. Do not add tests that require more than 2 GPUs — they will not fit in the CI resource budget.

### Functional Test Launcher Scripts

Functional tests are placed in tiered launcher scripts inside [`tests/functional_tests/`](tests/functional_tests/). Each tier runs in a separate CI job, allowing faster PR feedback while keeping thorough coverage on nightly runs.

| Tier | Prefix | Trigger | Purpose |
|------|--------|---------|---------|
| **L0** | `L0_Launch_*.sh` | Every PR, main push, schedule | Core smoke tests — must be fast and stable |
| **L1** | `L1_Launch_*.sh` | Main push + schedule; PRs labeled `needs-more-tests` | Broader model/recipe coverage |
| **L2** | `L2_Launch_*.sh` | Schedule / `workflow_dispatch` only | VL models, checkpoint conversion, heavy quantization |

When adding a new launcher script, always start with the **L0** tier so it runs on every PR. A maintainer will adjust the tier later if the test is too slow or better suited for nightly coverage. You must **also update** [`.github/workflows/cicd-main.yml`](.github/workflows/cicd-main.yml) to include it in the corresponding job matrix:

```yaml
# Example: adding an L1 test
- script: L1_Launch_your_new_test
```

Without this step, your new launcher script will not be picked up by CI.

## 📦 Dependencies Management

We use [uv](https://docs.astral.sh/uv/) for managing dependencies. For reproducible builds, our project tracks the generated `uv.lock` file in the repository.
On a weekly basis, the CI attempts an update of the lock file to test against upstream dependencies.

### Adding a New Dependency

**Adding required (non-optional) dependencies is strongly discouraged** and will be strictly reviewed. Every required dependency inflates the package and container image for all downstream consumers — most of whom will not need it. Prefer **optional dependencies** under an extra group whenever possible.

If your feature requires a dependency that is not already in `pyproject.toml`, **submit the dependency change as a separate PR first**. Do not bundle dependency additions with feature code — this keeps reviews focused and makes CI failures easier to diagnose.

1. Add the dependency to `pyproject.toml` (either via `uv add` or by editing the file directly):

```bash
# Preferred: optional dependency under an extra group
uv add --optional --extra $EXTRA $DEPENDENCY

# Required dependency (needs strong justification — affects all downstream)
uv add $DEPENDENCY
```

`EXTRA` refers to the subgroup of extra-dependencies to which you're adding the new dependency.
Example: For adding a TRT-LLM specific dependency, run `uv add --optional --extra trtllm $DEPENDENCY`.

2. Regenerate the lock file:

```bash
uv lock
```

3. Commit both files and open a PR:

```bash
git add pyproject.toml uv.lock
git commit -s -m "[build] chore: Add $DEPENDENCY"
git push
```

4. Once the dependency PR is merged, rebase your feature branch onto `main` and open the feature PR.

### 🧹 Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI does not auto-fix linting and formatting issues, but most issues can be fixed by running the following command:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Note: If `ruff` is missing, please follow the [installation](#local-workstation) guide.


## 📄 Documentation and Test Requirements

### All Features

**Every feature PR must evaluate whether documentation and tests need to be added or updated.** This applies to all changes, not just large features. Before submitting a PR, check:

- [ ] **Docs**: Does this change need a new doc page, or does an existing doc need updating?
- [ ] **Tests**: Does this change need new unit or functional tests, or do existing tests need updating?

For new key features (e.g., enabling a new model, enabling a new parallelism strategy), documentation is **required**. The documentation should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

This ensures that all significant changes are well-thought-out and properly documented for future reference. Comprehensive documentation serves two critical purposes:

1. **User Adoption**: Helps users understand how to effectively use the library's features in their projects
2. **Developer Extensibility**: Enables developers to understand the internal architecture and implementation details, making it easier to modify, extend, or adapt the code for their specific use cases

Quality documentation is essential for both the usability of Megatron-Bridge and its ability to be customized by the community.

### Refactoring PRs

Refactoring PRs that rename symbols, move files, or change paths are high risk for creating stale references. When a refactor changes any public name or path, you **must** check whether the following need corresponding updates:

- [ ] **Docs**: Any docs that reference the old names, paths, config keys, or CLI arguments
- [ ] **Docstrings**: Docstrings in the codebase that mention the old names or paths
- [ ] **Scripts**: Example and training scripts under `scripts/` that import or reference the old names or paths

A refactor PR that renames something without updating all references will break users silently. Reviewers should verify these are addressed before approving.

## ✨ Code Quality

- Follow the existing code style and conventions (see [skills/code-style/SKILL.md](skills/code-style/SKILL.md))
- Write tests for new features
- Update documentation to reflect your changes
- Ensure all tests pass before submitting a PR
- Do not add arbitrary defaults for configs, be as explicit as possible

## ✍️ Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

## 🚀 Running GitHub CI

There are two ways to trigger CI tests on your pull request:

### Automatic CI Triggering

If your GitHub user is configured to use [signed commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification), CI tests will run automatically when you push commits to your pull request.

> **Note**: Signed commits are different from signing-off on commits (which uses the `-s` flag mentioned in the [Signing Your Work](#signing-your-work) section).

### Manual CI Triggering

If you don't have signed commits set up, you can still trigger CI tests manually by commenting on your pull request:

```
/ok to test <commit-SHA>
```

For example:

```
/ok to test a1b2c3d4e5f6
```

**Important**: You'll need to add this comment for each new commit you push to ensure CI tests run on the latest changes.

#### Finding Your Commit SHA

You can find the commit SHA in several ways:

- View your pull request's commit history on GitHub
- Run `git log --oneline -1` in your local repository
- Check the commit details in your Git client

## 🤖 Contributing Models

Please see our [documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html) for a detailed guide on contributing new models.
