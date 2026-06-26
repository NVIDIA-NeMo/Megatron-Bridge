# Fork Maintenance Guide

This document explains how we maintain the `kaiko-ai/Megatron-Bridge` fork of `NVIDIA-NeMo/Megatron-Bridge`.

## Why we fork

We want kaiko-specific Bridge patches to live as real commits (not runtime monkey-patches in `libs/kbridge`), and we want to adopt newer Bridge fixes without waiting for the next NeMo container release. The fork is the home for both. The `kmbridge-nemo` Docker image in `kaiko-eng` installs Bridge from it. See [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) for the rationale.

## Policy

Fork `main` is **the latest Bridge release tag that's compatible with the current `kmbridge-nemo` base image, plus our patches on top**. Currently based on **`v0.4.2`**. When a newer compatible release ships, rebase main onto it.

Besides `main`, the fork mirrors upstream's release branches (e.g. `r0.4.0`) so we have a stable target to rebase main onto.

## First-time setup

```bash
git clone --recurse-submodules https://github.com/kaiko-ai/Megatron-Bridge.git
cd Megatron-Bridge
git remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Format/lint hooks Bridge uses upstream
pip install pre-commit
pre-commit install
```

After this:

- `origin` = our fork, `upstream` = NVIDIA-NeMo.
- Megatron-Core is vendored at `3rdparty/Megatron-LM` and initialized by `--recurse-submodules`. We do not fork Core separately — it follows Bridge's submodule pin. If you cloned without the flag, run `git submodule update --init`.
- Every `git commit` runs ruff format / lint and the other hooks Bridge uses upstream.

## Workflow: adding a kaiko change

1. Branch off `main`: `git checkout -b feat/short-description main`.
2. Make the change. Keep it small.
3. Commit with sign-off using Bridge's `[area] type: description` format per their [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md), e.g. `git commit -s -m "[training] feat: add step-0 validation flag"`.
4. Open a PR targeting `main`. Apply the labels from [Repo configuration](#repo-configuration); for VL-touching changes, also apply `full-test-suite`.
5. After internal review and squash-merge, bump `MEGATRON_BRIDGE_SHA` in kaiko-eng's `kmbridge-nemo` Dockerfile and ship a new image version.
6. When ready, submit a PR to upstream (`NVIDIA-NeMo/Megatron-Bridge`).

### Forgot to sign off?

NVIDIA rejects unsigned commits. Fix the whole branch in one go:

```bash
git rebase --signoff main
git push --force-with-lease
```

## Testing

Upstream's unit-test job lives on NVIDIA's self-hosted runners and only fires for PRs against `NVIDIA-NeMo/Megatron-Bridge`, so it never runs on our PRs. To get signal on fork PRs we run our own lightweight job — `.github/workflows/kaiko-tests.yml`, which fires on every PR against `main` and runs `tools/ci/run_tests.sh`. That same script runs locally to reproduce a result (see below).

How it works:

- **Regression-diff, not pass/fail.** The job runs the full CPU unit suite against *two* checkouts — the PR branch and the current `main` — and fails only on tests that fail on the PR but pass on `main`. Thus, we only care about regressions the change introduces.
- **Image source.** It pulls the NeMo image from our ghcr mirror (`ghcr.io/kaiko-ai/nvidia-nemo-mirror:<tag>`) rather than nvcr, for self-contained pulls. The fork is public, so it can't use kaiko runners or the private `kmbridge-nemo` image — but the public `nemo` image has the same torch/TE/Megatron-Core stack the unit tests need.
- **Maintaining the mirror.** `.github/workflows/mirror-nemo.yml` copies `nvcr.io/nvidia/nemo:<tag>` into ghcr. It's `workflow_dispatch`-only: **run it once whenever you introduce a new tag** (e.g. when bumping the `IMAGE` tag in `kaiko-tests.yml`). Add the new tag to that workflow's matrix and trigger it from the Actions tab.


### Running it locally

`run_tests.sh` needs the test image and two checkouts — `main` and your branch. From your branch, add a sibling checkout of `main` and point the script at both:

```bash
git worktree add ../main origin/main        # one-time: a sibling checkout of main
IMAGE=nvcr.io/nvidia/nemo:26.06 tools/ci/run_tests.sh ../main .
```

It runs the suite inside the image with each checkout overlaid via `PYTHONPATH` (so `import megatron.bridge.*` resolves to that checkout, while `megatron.core`, `torch`, and TE come from the image — no submodule init needed), CPU-only, against a read-only mount. On Apple Silicon the image runs under amd64 emulation: the "CPU does not support AVX → illegal instruction likely" boot warning is harmless — `torch` still imports (slowly; a few minutes, then the tests run in seconds).

## Workflow: rebasing onto a newer Bridge release

When upstream cuts a release we want to move to:

```bash
git fetch upstream <release-branch>          # e.g. r0.4.0 or r0.5.0
git checkout <release-branch>
git reset --hard upstream/<release-branch>
git push --force-with-lease origin <release-branch>

git checkout main
git rebase --onto <new-tag> <old-tag> main   # e.g. --onto v0.4.3 v0.4.2 main
git push --force-with-lease origin main
```

Then bump `MEGATRON_BRIDGE_SHA` in kaiko-eng and ship a new image version.

## Cadence expectations

Upstream PRs from external contributors land in **weeks, not days** — NVIDIA's CI requires a maintainer to approve every run, and VL-touching changes need the heavy `full-test-suite` tier.

## Repo configuration

- **Branch protection.** `main` requires PR + 1 approval, no force-push, no deletion. Release-track branches (`r0.4.0`, future ones) allow force-push (the rebase workflow needs it) but forbid deletion.
- **Team access.** `@kaiko-ai/MLLM` has `write`. The CODEOWNERS team mention resolves through this grant.
- **Labels.** `area:training`, `feature`, `bug`, `docs`, `full-test-suite` mirror upstream Bridge — apply them on our PRs the same way you would upstream.
- **Merge style.** Auto-merge and auto-delete-branch-on-merge are enabled. Squash commits take the PR title as the subject and the PR body as the message.

## See also

- [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) — fork policy and principles
- [kaiko-ai/kaiko-eng#33272](https://github.com/kaiko-ai/kaiko-eng/issues/33272) — initial setup (this fork)
- Upstream: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- Upstream contributing guide: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md
