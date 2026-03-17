# Git Commit & PR Workflow for Megatron-Hub

## Overview

When the user asks to commit, push, or submit changes, follow this full workflow automatically.

## Workflow Steps

### Step 1: Check Current Branch

```bash
git branch --show-current
```

- If on `main`, **always create a new branch** before committing.
- Use a descriptive branch name based on the changes (e.g., `yuya/fix-qwen3-conversion`, `yuya/add-fim-dataset-support`).

```bash
# Example: create and switch to a new branch
git checkout -b yuya/<descriptive-branch-name>
```

### Step 2: Stage Changes and Run Pre-commit

```bash
# Check what has changed
git status
git diff --stat

# Stage all updated changes
git add -u

# Or stage specific files if the user requests
git add <file1> <file2> ...

# Ensure pre-commit is on PATH
export PATH="/Users/yuya/Library/Python/3.9/bin:$PATH"

# Run pre-commit hooks on staged files to fix lint issues
pre-commit run
```

- Pre-commit runs ruff (linting + formatting), end-of-file-fixer, and trailing-whitespace checks.
- If pre-commit auto-fixes files, **re-stage** the fixed files and re-run until it passes:

```bash
# If pre-commit modified files, re-stage and re-run
git add -u
pre-commit run
```

- Repeat until all hooks pass. If any issues cannot be auto-fixed, manually fix them before proceeding.

### Step 3: Commit Changes (Signed)

```bash
# Commit with a descriptive message and sign-off
git commit -s -m "<descriptive commit message>"
```

- Always use `-s` flag to sign off the commit (`Signed-off-by` line).
- The sign-off identity is: `Yu Yao <yaoyu.094@gmail.com>`. If git `-s` produces a different name/email, use `--trailer "Signed-off-by: Yu Yao <yaoyu.094@gmail.com>"` instead.
- Write clear, concise commit messages describing what changed and why.
- Use conventional commit style when appropriate (e.g., `fix:`, `feat:`, `refactor:`, `test:`).

### Step 4: Push to Remote

```bash
# Push the branch to remote (set upstream on first push)
git push -u origin <branch-name>

# Subsequent pushes
git push
```

### Step 5: Create a PR (if one doesn't already exist)

```bash
# Check if a PR already exists for the current branch
gh pr list --head <branch-name>

# If no PR exists, create one
gh pr create --title "<PR title>" --body "<PR description>" --base main
```

- Use `gh` CLI to create the PR.
- PR title must follow Megatron-Bridge format from `CONTRIBUTING.md`:
  - `[{modules}] {type}: {description}`
  - Modules examples: `model`, `recipe`, `training`, `data`, `ckpt`, `peft`, `perf`, `ci`, `doc`, `test`, `build`, `misc`
  - Types: `feat`, `fix`, `refactor`, `chore`, `test`
  - For API/signature/CLI/config breaking changes, prefix with `[BREAKING]` (e.g., `[BREAKING][training] refactor: Change optimizer config structure`)
- Body should summarize the changes made.

### Step 6: Comment `/ok to test` on the PR

After pushing, comment on the PR to trigger CI testing:

```bash
# Get the latest commit hash
COMMIT_HASH=$(git rev-parse HEAD)

# Get the PR number
PR_NUMBER=$(gh pr list --head <branch-name> --json number --jq '.[0].number')

# Comment to trigger testing
gh pr comment "$PR_NUMBER" --body "/ok to test $COMMIT_HASH"
```

## Complete Workflow Example

```bash
# 1. Check branch
CURRENT_BRANCH=$(git branch --show-current)

# 2. Create new branch if on main
if [ "$CURRENT_BRANCH" = "main" ]; then
  git checkout -b yuya/<descriptive-name>
fi

# 3. Stage changes
git add -u

# 4. Ensure pre-commit is on PATH, run pre-commit, and re-stage until clean
export PATH="/Users/yuya/Library/Python/3.9/bin:$PATH"
pre-commit run
git add -u
pre-commit run

# 5. Commit (signed)
git commit -s -m "feat: add support for new feature"

# 6. Push
git push -u origin $(git branch --show-current)

# 7. Create PR if needed
BRANCH=$(git branch --show-current)
EXISTING_PR=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number')
if [ -z "$EXISTING_PR" ]; then
  gh pr create --title "Title" --body "Description" --base main
fi

# 8. Comment /ok to test
PR_NUMBER=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number')
COMMIT_HASH=$(git rev-parse HEAD)
gh pr comment "$PR_NUMBER" --body "/ok to test $COMMIT_HASH"
```

## Important Notes

- **Never commit directly to `main`** — always create a feature branch.
- **Always run pre-commit before committing** — fix all lint issues first.
- **Always sign commits with `-s`** — every commit must include `Signed-off-by`.
- **Always push before creating a PR** — the remote branch must exist.
- **Always format PR titles as `[{modules}] {type}: {description}`** — follow the module/type list from `CONTRIBUTING.md`.
- **Use `[BREAKING]` prefix for breaking API/CLI/config/signature changes**.
- **Always comment `/ok to test <commit_hash>`** after pushing to trigger CI.
- Use `git_write` permissions when running git commands and `network` permissions when pushing or using `gh`.
- **When the user asks to commit, always push and comment `/ok to test` automatically** — do not ask for confirmation to push or comment; just run the full workflow end-to-end.
- Use `required_permissions: ["all"]` for `gh` CLI commands to avoid TLS certificate issues in the sandbox.
