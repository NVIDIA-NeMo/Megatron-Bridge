# Git Commit Workflow

## Standard Workflow for Committing Changes

### 1. Check Current Branch
```bash
git branch --show-current
```

### 2. Create Feature Branch (REQUIRED if on main)
**Important**: You must create a feature branch before committing if you're on `main`. Never commit directly to `main`.

Check your current branch:
```bash
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" = "main" ]; then
    git checkout -b feature/your-feature-name
fi
```

Or manually:
```bash
git checkout -b feature/your-feature-name
```

### 3. Stage Your Changes
```bash
git add <file1> <file2> ...
# Or stage all modified files:
git add -u
```

### 4. Run Pre-commit Hooks
```bash
export PATH="/Users/yuya/Library/Python/3.9/bin:$PATH"
pre-commit run
```
This will run all pre-commit hooks (formatting, linting, etc.) before committing.

### 5. Commit with Sign-off
```bash
git commit -s -m "[module] type: Your descriptive commit message"
```
The `-s` flag adds a Signed-off-by line to the commit message. See [Commit and PR Title Format](#commit-and-pr-title-format) for proper formatting.

### 6. Push to Remote
```bash
git push
```

### 7. Check for Existing PR
```bash
gh pr list --head <your-branch-name> --json number --jq '.[0].number'
```

### 8. Trigger CI Testing
If a PR exists, comment on it to trigger CI:
```bash
COMMIT_HASH=$(git rev-parse HEAD)
gh pr comment <PR_NUMBER> --body "/ok to test $COMMIT_HASH"
```

## Example Workflow

```bash
# 1. Check branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# 2. Create feature branch if on main (REQUIRED)
if [ "$CURRENT_BRANCH" = "main" ]; then
    git checkout -b feature/your-feature-name
fi

# 3. Stage changes
git add tests/unit_tests/models/gemma_vl/test_gemma3_vl_bridge.py

# 4. Run pre-commit
export PATH="/Users/yuya/Library/Python/3.9/bin:$PATH"
pre-commit run

# 5. Commit with sign-off
git commit -s -m "[test] fix: Fix gemma3_vl bridge test for image_token_id default"

# 6. Push
git push

# 7. Check for PR and trigger CI
PR_NUMBER=$(gh pr list --head feature/provider-bridge-refactor-3 --json number --jq '.[0].number')
COMMIT_HASH=$(git rev-parse HEAD)
gh pr comment $PR_NUMBER --body "/ok to test $COMMIT_HASH"
```

## Commit and PR Title Format

Format your commit messages and PR titles as:

```text
[{modules}] {type}: {description}
```

### Modules
Use the most relevant ones, separate multiple with `,`:
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

### Types
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code refactoring without changing functionality
- `chore` - Maintenance tasks
- `test` - Adding or updating tests

### Breaking Changes
If your PR breaks any API (CLI arguments, config, function signature, etc.), add `[BREAKING]` to the beginning of the title.

### Examples
```text
[model] feat: Add Qwen3 model bridge
[recipe, doc] feat: Add Llama 3.1 70B recipe with documentation
[ckpt] fix: Handle missing keys in HF checkpoint conversion
[BREAKING][training] refactor: Change optimizer config structure
[ci, build] chore: Update ruff version
[test] fix: Fix gemma3_vl bridge test for image_token_id default
```

## Notes

- **Never commit directly to `main`** - Always create a feature branch first
- Always run `pre-commit run` before committing to catch formatting/linting issues early
- Use descriptive commit messages following the format above
- The `-s` flag is required for DCO (Developer Certificate of Origin) compliance
- If pre-commit modifies files, you may need to stage them again before committing
