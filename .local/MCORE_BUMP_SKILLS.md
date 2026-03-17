# Megatron-Core (mcore) Submodule Bump Workflow

## Overview

Bump the `3rdparty/Megatron-LM` submodule to the latest commits from the upstream `main` and `dev` branches. This involves updating pinned hashes in `scripts/switch_mcore.sh`, checking out the submodule to the desired commit, and regenerating `uv.lock`.

## Prerequisites

- **Local submodule remote**: `nvidia` → `https://github.com/NVIDIA/Megatron-LM.git`
- **Workstation submodule remote**: `origin` → `git@github.com:NVIDIA/Megatron-LM.git` (different name!)
- SSH access to workstation via `ssh ws`
- Docker image `megatron-bridge` available on the workstation
- Local project path: `/Users/yuya/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/projects/Megatron-Hub/`
- Workstation project path: `/home/yuya/Projects/Megatron-Bridge`

## Step 1: Fetch Latest Commits

```bash
cd 3rdparty/Megatron-LM
git fetch nvidia main dev
```

Get the latest commit hashes:

```bash
# Main branch
git rev-parse nvidia/main

# Dev branch
git rev-parse nvidia/dev
```

## Step 2: Update `scripts/switch_mcore.sh`

Update both pinned commit hashes at the top of the file:

```bash
MAIN_COMMIT="<new-main-hash>"   # main branch
DEV_COMMIT="<new-dev-hash>"     # dev branch
```

## Step 3: Checkout the Submodule

Checkout the submodule to the target commit (typically dev for active development):

```bash
cd 3rdparty/Megatron-LM
git checkout <target-commit-hash>
```

Verify:

```bash
git log --oneline -1
```

## Step 4: Stage the Submodule Change

```bash
# From repo root
git add 3rdparty/Megatron-LM
git add scripts/switch_mcore.sh
```

## Step 5: Update `uv.lock`

The `uv.lock` file must be regenerated because `megatron-core` is a path dependency (`3rdparty/Megatron-LM/`). This **cannot** be done on the local Mac — it requires a Linux environment with CUDA and the full dependency stack.

### 5a. Update submodule on workstation directly (preferred)

**Do NOT rsync the full `3rdparty/Megatron-LM` directory** — it contains thousands of files and rsync will time out or disconnect mid-transfer. Instead, update the submodule directly on the workstation via git:

```bash
ssh ws "cd /home/yuya/Projects/Megatron-Bridge/3rdparty/Megatron-LM && \
  git fetch origin dev 2>&1 && \
  git checkout -f <target-commit-hash> 2>&1 && \
  git clean -fd 2>&1 && \
  git log --oneline -1"
```

**Why `-f` and `clean`?** The workstation's submodule often has local modifications or untracked files from previous work. A plain `git checkout` will fail with "Your local changes would be overwritten". Use `git checkout -f` to force, then `git clean -fd` to remove untracked files.

**Note**: The workstation's submodule remote is named `origin` (not `nvidia` like local). Always use `origin` when fetching on the workstation.

### 5a (alternative). Rsync if submodule update via git is not possible

Only if the workstation submodule is in a broken state and git operations fail:

```bash
rsync -avz --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='docs/_build/' \
  --exclude='.gitignore' --exclude='.gitattributes' \
  "/Users/yuya/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/projects/Megatron-Hub/" \
  ws:/home/yuya/Projects/Megatron-Bridge
```

**Warning**: This will take a long time and may disconnect. Do NOT exclude `3rdparty` when syncing for mcore bump.

### 5b. Regenerate uv.lock inside Docker on workstation

```bash
ssh ws "cd /home/yuya/Projects/Megatron-Bridge && \
  docker run --gpus all --rm \
    -v /home/yuya:/home/yuya \
    -v /home/yuya/Projects/Megatron-Bridge:/opt/Megatron-Bridge \
    --ipc=host \
    megatron-bridge \
    bash -c 'cd /opt/Megatron-Bridge && uv lock'"
```

**Do NOT use `-it` flag** when running docker via ssh — there is no interactive TTY and the command will hang or fail. Use `--rm` only.

### 5c. Copy updated uv.lock back to local

```bash
scp ws:/home/yuya/Projects/Megatron-Bridge/uv.lock \
  "/Users/yuya/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/projects/Megatron-Hub/uv.lock"
```

## Step 6: Commit and Push

Follow the standard commit workflow from `GIT_COMMIT_SKILLS.md`:

```bash
git add -u
git add 3rdparty/Megatron-LM

export PATH="/Users/yuya/Library/Python/3.9/bin:$PATH"
pre-commit run
git add -u
pre-commit run

git commit -s -m "chore: bump mcore submodule to latest main/dev"
git push -u origin $(git branch --show-current)
```

## Quick Reference — Full Workflow

```bash
# 1. Fetch latest (LOCAL)
cd 3rdparty/Megatron-LM
git fetch nvidia main dev
MAIN_HASH=$(git rev-parse nvidia/main)
DEV_HASH=$(git rev-parse nvidia/dev)
echo "Main: $MAIN_HASH"
echo "Dev:  $DEV_HASH"
cd ../..

# 2. Update scripts/switch_mcore.sh with MAIN_HASH and DEV_HASH

# 3. Checkout submodule locally to dev (or main)
cd 3rdparty/Megatron-LM
git checkout $DEV_HASH
cd ../..

# 4. Update submodule on workstation via git (NOT rsync)
ssh ws "cd /home/yuya/Projects/Megatron-Bridge/3rdparty/Megatron-LM && \
  git fetch origin dev && \
  git checkout -f $DEV_HASH && \
  git clean -fd && \
  git log --oneline -1"

# 5. Regenerate uv.lock in Docker (no -it flag!)
ssh ws "cd /home/yuya/Projects/Megatron-Bridge && \
  docker run --gpus all --rm \
    -v /home/yuya:/home/yuya \
    -v /home/yuya/Projects/Megatron-Bridge:/opt/Megatron-Bridge \
    --ipc=host \
    megatron-bridge \
    bash -c 'cd /opt/Megatron-Bridge && uv lock'"

# 6. Copy uv.lock back
scp ws:/home/yuya/Projects/Megatron-Bridge/uv.lock ./uv.lock

# 7. Stage, pre-commit, commit, push
git add -u && git add 3rdparty/Megatron-LM
export PATH="/Users/yuya/Library/Python/3.9/bin:$PATH"
pre-commit run && git add -u && pre-commit run
git commit -s -m "chore: bump mcore submodule to latest main/dev"
git push -u origin $(git branch --show-current)
```

## Notes

- The `uv.lock` regeneration must happen in the Docker container because many dependencies (e.g., `transformer-engine`, `mamba-ssm`, `flash-attn`) require CUDA and Linux.
- Always update **both** `MAIN_COMMIT` and `DEV_COMMIT` in `switch_mcore.sh`, even if only switching to one.
- After bumping, run `./scripts/switch_mcore.sh status` to verify the pinned hashes match.
- If the bump introduces breaking API changes in mcore, additional code changes in `src/megatron/bridge/` may be needed.

## Troubleshooting

### Issue: rsync of 3rdparty/Megatron-LM times out or disconnects

**Symptom**: `rsync` exits with code 12 ("error in rsync protocol data stream") or hangs indefinitely when syncing the full repo including `3rdparty/Megatron-LM`.

**Cause**: The Megatron-LM submodule contains thousands of files (tests, golden values, docs, etc.) and the transfer is too large for a single rsync over SSH.

**Fix**: Do NOT rsync the submodule. Instead, update it directly on the workstation via git:

```bash
ssh ws "cd /home/yuya/Projects/Megatron-Bridge/3rdparty/Megatron-LM && \
  git fetch origin dev && \
  git checkout -f <commit-hash> && \
  git clean -fd"
```

### Issue: `git checkout` fails on workstation with "local changes would be overwritten"

**Symptom**: `git checkout <hash>` in the workstation's submodule fails with a long list of modified/untracked files that would be overwritten.

**Cause**: Previous work, other branches, or Docker container writes left dirty state in the submodule working tree.

**Fix**: Use force checkout and clean:

```bash
git checkout -f <commit-hash>
git clean -fd
```

The `-f` flag forces checkout despite local modifications. `git clean -fd` removes untracked files and directories. This is safe because the submodule should always be at a pinned upstream commit, not contain local work.

### Issue: Submodule remote name differs between local and workstation

**Symptom**: `git fetch nvidia dev` works locally but fails on the workstation (or vice versa).

**Cause**: The upstream remote has different names:
- **Local**: `nvidia` → `https://github.com/NVIDIA/Megatron-LM.git`
- **Workstation**: `origin` → `git@github.com:NVIDIA/Megatron-LM.git`

**Fix**: Use the correct remote name per environment. Check with `git remote -v` if unsure:

```bash
# Local
git fetch nvidia main dev

# Workstation
ssh ws "cd /home/yuya/Projects/Megatron-Bridge/3rdparty/Megatron-LM && git fetch origin main dev"
```

### Issue: Docker command hangs when run via SSH

**Symptom**: `docker run -it ...` via `ssh ws "..."` hangs indefinitely or produces garbled output.

**Cause**: The `-it` flag allocates a pseudo-TTY and interactive stdin, but there is no real TTY when running through `ssh "command"`.

**Fix**: Remove the `-it` flag. Use only `--rm` for non-interactive one-shot commands:

```bash
# Wrong (hangs via ssh)
docker run --gpus all -it --rm ... bash -c 'uv lock'

# Correct
docker run --gpus all --rm ... bash -c 'uv lock'
```

If you need interactive access, use `ssh -t ws "docker run -it ..."` (the `-t` flag on ssh forces TTY allocation).

### Issue: `uv lock` fails with resolution errors

**Symptom**: `uv lock` inside Docker fails with dependency resolution errors or missing packages.

**Cause**: The new mcore commit may have changed its `pyproject.toml` dependencies, introducing conflicts with Megatron-Bridge's dependency tree.

**Fix**:
1. Check the mcore `pyproject.toml` diff between old and new commits for dependency changes.
2. Update Megatron-Bridge's `pyproject.toml` to resolve conflicts.
3. Re-run `uv lock`.

```bash
# Compare mcore dependency changes
cd 3rdparty/Megatron-LM
git diff <old-commit>..<new-commit> -- pyproject.toml
```
