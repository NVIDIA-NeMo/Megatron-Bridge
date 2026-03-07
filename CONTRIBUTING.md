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
[{modules}] {type}: {description}
```

**Modules** (use the most relevant ones, separate multiple with `,`):
- `model` - Model implementations and bridges
- `recipe` - Training recipes
- `training` - Training loop and utilities
- `data` - Data loading and processing
- `ckpt` - Checkpoint conversion and saving
- `peft` - Parameter-efficient fine-tuning (LoRA, etc.)
- `perf` - Performance optimizations
- `ci` - CI/CD configuration
- `doc` - Documentation
- `test` - Tests
- `build` - Build system and dependencies
- `misc` - Other changes

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
[recipe, doc] feat: Add Llama 3.1 70B recipe with documentation
[ckpt] fix: Handle missing keys in HF checkpoint conversion
[BREAKING][training] refactor: Change optimizer config structure
[ci, build] chore: Update ruff version
```

## 📝 Writing Tests

We use [pytest](https://docs.pytest.org/en/stable/) for writing both unit and functional tests.

**Unit tests** aim to test functions in isolation. They generally do not depend on artifacts like Hugging Face checkpoints or larger datasets. Exception to this is a small toy dataset consisting of tokenizers.
Unit tests are stored at `tests/unit_tests`. Please add your test to an existing folder or create a new one if none matches.

**Functional tests** are integration tests that perform model training or operate on larger artifacts. We use pytest for writing these. In some cases, it might be desired to run your test (or parts of it) in a subprocess to avoid process contamination. We use `subprocess.run` for this inside the pytest function. Please add your test into one of the predefined folders. If none of the folders matches semantically, please reach out to the `@nvidia-nemo/automation` in your PR for consultation.

### Functional Test Launcher Scripts

Functional tests that take longer to run should be placed in a `L2_Launch_*.sh` launcher script inside the [`tests/functional_tests/`](tests/functional_tests/) folder. These launcher scripts allow CI to run test groups in parallel, significantly reducing overall pipeline time.

When adding a new `L2_Launch_*.sh` file, you **must** also update [`.github/workflows/cicd-main.yml`](.github/workflows/cicd-main.yml) to include it in the `cicd-functional-tests` job matrix. Add a new entry under `matrix.include`, for example:

```yaml
- script: L2_Launch_your_new_test
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
