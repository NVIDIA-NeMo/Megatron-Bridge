# MCore Submodule Bump

Bump the `3rdparty/Megatron-LM` submodule to the latest upstream commits.
An automated daily job creates bump PRs titled
`chore(beep boop): bump mcore submodule ...`; the maintainer monitors and fixes
failures.

---

## Key Invariant

The submodule pointer committed to the repo **must always point to the main
commit**, not dev. The `uv.lock` in the repo is generated against the main
MCore commit. Dev is only used for CI variant testing.

---

## Procedure

### Step 1: Fetch latest commits

Fetch the latest `main` and `dev` branch heads from the upstream Megatron-LM
repo within the submodule.

### Step 2: Update pinned hashes

Update the commit hashes in the two files at the repo root:

- `.main.commit` — pinned main branch commit
- `.dev.commit` — pinned dev branch commit

These are read by `scripts/switch_mcore.sh` and by CI.

### Step 3: Checkout the submodule

Checkout `3rdparty/Megatron-LM` to the new **main** commit.

### Step 4: Regenerate `uv.lock`

Because `megatron-core` is a path dependency (`3rdparty/Megatron-LM/`),
`uv.lock` must be regenerated whenever the submodule changes. This requires a
Linux environment with CUDA and the full dependency stack — it cannot be done on
a Mac.

Run `uv lock` inside the appropriate container or environment, then copy the
updated `uv.lock` back.

### Step 5: Commit and push

Stage the submodule change, `.main.commit`, `.dev.commit`, and `uv.lock`. Commit
with a conventional message:

```
chore: bump mcore submodule to latest main/dev
```

---

## CI Variant Testing

The repo supports testing against both main and dev MCore commits in CI:

- **`main`** (default): Uses the submodule as-is with `uv sync --locked`
- **`dev`**: Overrides the submodule to `DEV_COMMIT`, builds a separate
  container without `--locked`
- **`both`**: Runs the full pipeline twice, once for each variant

Both `.main.commit` and `.dev.commit` must be kept up to date even though only
main is committed as the submodule pointer.

---

## When the Automated Bump PR Fails

The daily automated bump creates a PR. When its CI fails:

1. **Do NOT push fixes to the automated bump branch.** Instead, cherry-pick the
   bump commit into a new fix branch off `main`.

2. Apply the fix (update Bridge code, fix imports, adapt to API changes).

3. Open a fix PR that references the original bump PR.

4. Close the original bump PR (or let the `Closes #` keyword handle it on
   merge).

5. Land the fix PR as quickly as possible to avoid falling behind upstream.

---

## Troubleshooting

### `uv lock` fails with resolution errors

The new MCore commit may have changed its `pyproject.toml` dependencies. Compare
the old and new MCore `pyproject.toml` to identify dependency changes, then
update Bridge's `pyproject.toml` to resolve conflicts.

### Submodule has dirty state

If the submodule working tree has local modifications or untracked files, use
force checkout and clean:

```bash
cd 3rdparty/Megatron-LM
git checkout -f <commit-hash>
git clean -fd
```
