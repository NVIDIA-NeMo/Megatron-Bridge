# Contributing To Megatron-Bridge

Thanks for your interest in contributing to Megatron-Bridge!

## 🛠️ Setting Up Your Environment

### Development Environment

You can either follow the steps below to set up the environment from scratch, or use the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags), which provides a pre-built environment and makes these steps unnecessary.

**Build and run the Docker container**:

First, set up the third-party modules of this repository so they can be copied into the container:

```bash
git submodule update --init --recursive
```

Build the container:

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

Use at most one state label from this set at a time:

| Label | Meaning |
| --- | --- |
| `needs-triage` | New item needs classification and ownership |
| `needs-review` | PR is ready for code review and waiting on a reviewer |
| `needs-author` | Author action is required before review or merge can continue |
| `needs-follow-up` | Issue or PR has finished initial triage/review and needs further follow-up |
| `blocked` | Work cannot move forward until an external dependency is cleared |
| `ready-to-merge` | PR is approved, current, and only waiting for CI to pass before merge |

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

**Functional tests** are integration tests that perform model training or operate on larger artifacts. We use pytest for writing these. In some cases, it might be desired to run your test (or parts of it) in a subprocess to avoid process contamination. We use `subprocess.run` for this inside the pytest function. Please add your test into one of the predefined folders. If none of the folders matches semantically, please reach out to the `@nvidia-nemo/automation` in your PR for consultation.

### Functional Test Launcher Scripts

Functional tests are placed in tiered launcher scripts inside [`tests/functional_tests/`](tests/functional_tests/). Each tier runs in a separate CI job, allowing faster PR feedback while keeping thorough coverage on nightly runs.

| Tier | Prefix | Trigger | Purpose |
|------|--------|---------|---------|
| **L0** | `L0_Launch_*.sh` | Every PR, main push, schedule | Core smoke tests — must be fast and stable |
| **L1** | `L1_Launch_*.sh` | Main push + schedule (not PRs) | Broader model/recipe coverage |
| **L2** | `L2_Launch_*.sh` | Schedule / `workflow_dispatch` only | VL models, checkpoint conversion, heavy quantization |

When adding a new launcher script, choose the appropriate tier and **also update** [`.github/workflows/cicd-main.yml`](.github/workflows/cicd-main.yml) to include it in the corresponding `cicd-functional-tests-l{0,1,2}` job matrix:

```yaml
# Example: adding an L1 test
- script: L1_Launch_your_new_test
```

Without this step, your new launcher script will not be picked up by CI.

## 📦 Dependencies Management

We use [uv](https://docs.astral.sh/uv/) for managing dependencies. For reproducible builds, our project tracks the generated `uv.lock` file in the repository.
On a weekly basis, the CI attempts an update of the lock file to test against upstream dependencies.

New required dependencies can be added by `uv add $DEPENDENCY`.

New optional dependencies can be added by `uv add --optional --extra $EXTRA $DEPENDENCY`.

`EXTRA` refers to the subgroup of extra-dependencies to which you're adding the new dependency.
Example: For adding a TRT-LLM specific dependency, run `uv add --optional --extra trtllm $DEPENDENCY`.

Alternatively, the `pyproject.toml` file can also be modified directly.

Adding a new dependency will update UV's lock-file. Please check this into your branch:

```bash
git add uv.lock pyproject.toml
git commit -m "build: Adding dependencies"
git push
```

### 🧹 Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI does not auto-fix linting and formatting issues, but most issues can be fixed by running the following command:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Note: If `ruff` is missing, please follow the [installation](#local-workstation) guide.


## 📄 Documentation Requirement

**Important**: All new key features (e.g., enabling a new model, enabling a new parallelism strategy) must include documentation update (either a new doc or updating an existing one). This document update should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

This ensures that all significant changes are well-thought-out and properly documented for future reference. Comprehensive documentation serves two critical purposes:

1. **User Adoption**: Helps users understand how to effectively use the library's features in their projects
2. **Developer Extensibility**: Enables developers to understand the internal architecture and implementation details, making it easier to modify, extend, or adapt the code for their specific use cases

Quality documentation is essential for both the usability of Megatron-Bridge and its ability to be customized by the community.

## ✨ Code Quality

- Follow the existing code style and conventions (see [CODING_GUIDELINES.md](CODING_GUIDELINES.md))
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
