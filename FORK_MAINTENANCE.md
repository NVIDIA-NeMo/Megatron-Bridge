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
