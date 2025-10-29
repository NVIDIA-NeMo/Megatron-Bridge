# Contributing To Megatron-Bridge

Thanks for your interest in contributing to Megatron-Bridge!

## 🛠️ Setting Up Your Environment

You can either follow the steps below to set up the environment from scratch, or use the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags), which provides a pre-built environment and makes these steps unnecessary.

### Development Container

For containerized development, use our Dockerfile for building your own container:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t megatron-bridge \
    .
```

Start your container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  megatron-bridge
```

## 📝 Writing tests

We use [pytest](https://docs.pytest.org/en/stable/) for writing both unit and functional tests.

Unit tests aim to test functions in isolation. They generally do not depend on artifacts like Hugging Face checkpoints or larger datasets. Exception to this is a small toy dataset consisting of tokenizers.  
Unit tests are stored at `tests/unit_tests`. Please add your test to an existing folder or create a new one if no one matches.

Functional tests are integration tests that perform model training or operate on larger artifacts. We use pytest for writing these. In some cases, it might be desired to run your test (or parts of it) in a subprocess to avoid process contamination. We use `subprocess.Run` for this inside the pytest function. Please add your test into one of the predefined folders. If none of the folders matches semantically, please reach out to the `@nvidia-nemo/automation` in your PR for consultation.

## 📦 Dependencies management

We use [uv](https://docs.astral.sh/uv/) for managing dependencies. For reproducible builds, our project tracks the generated `uv.lock` file in the repository.  
On a weekly basis, the CI attemps an update of the lock file to test against upstream dependencies.

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

### 📝 Documentation

**Important**: All new key features (ex: enabling a new inference optimized library, enabling a new deployment option) must include documentation update (either a new doc or updating an existing one). This document update should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

This ensures that all significant changes are well-thought-out and properly documented for future reference. Comprehensive documentation serves two critical purposes:

1. **User Adoption**: Helps users understand how to effectively use the library's features in their projects
2. **Developer Extensibility**: Enables developers to understand the internal architecture and implementation details, making it easier to modify, extend, or adapt the code for their specific use cases

Quality documentation is essential for both the usability of Megatron-Bridge and its ability to be customized by the community.

## ✨ Code Quality

- Follow the existing code style and conventions
- Write tests for new features
- Update documentation to reflect your changes
- Ensure all tests pass before submitting a PR
- Do not add arbitrary defaults for configs, be as explicit as possible.

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

## Contributing Models

Please see our [documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html) for a detailed guide on contributing new models.
