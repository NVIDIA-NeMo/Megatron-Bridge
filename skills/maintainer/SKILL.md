---
name: maintainer
description: Maintainer and oncall utilities for Megatron Bridge. Covers issue triage, PR labeling, PR review with confidence scoring, and report generation. Use when the user says "triage", "label PRs", "review PRs", "PR sweep", "oncall", or asks to handle open issues or PRs.
---

# Maintainer Utilities

Internal workflows for Megatron Bridge maintainers and oncall. Not intended
for regular contributors.

This skill dispatches to two workflows based on what the user asks:

| User says | Workflow |
|---|---|
| "triage", "triage issues", "oncall triage" | [Issue triage](triage-workflow.md) |
| "label PRs", "tag PRs", "review PRs", "PR sweep", "deep review" | [PR label & review](pr-review-workflow.md) |
| "label and triage" | Both workflows |

## Shared Setup

All workflows use the same repo, permissions, and label taxonomy.

**Repo:**

```bash
NVIDIA-NeMo/Megatron-Bridge
```

**Permissions:** Use `required_permissions: ["all"]` for all `gh` commands.

**Before starting any workflow:**

1. Read the label taxonomy at [label-taxonomy.md](label-taxonomy.md).
2. Read the code-style skill at `skills/code-style/SKILL.md` for coding guideline rules.
3. Read any files in `reports/review-memory/*.md` if they exist.

## Author Affiliation Check

Used by both workflows. Check whether the author is internal:

```bash
gh api repos/NVIDIA-NeMo/Megatron-Bridge/collaborators/<AUTHOR>/permission --jq '.permission'
```

If not a collaborator (writer/admin/maintainer), treat as external and keep or add `community-request`.

## Labeling Rules

Both issues and PRs follow the same taxonomy from [label-taxonomy.md](label-taxonomy.md):

- **One type label** per item (`bug`, `feature`, `support`, `docs`, `ci`)
- **One primary area label** per item (use file-to-area mapping for PRs)
- **State labels** — at most one primary; `needs-author` + `needs-follow-up` can co-exist
- **Preserve orthogonal labels** (`community-request`, release labels, partner labels)

**PR-specific notes:**
- `support` is rarely used for PRs; prefer `bug`, `feature`, `docs`, or `ci`.
- For mixed PRs, choose the dominant area.
- Do not use `needs-triage` on PRs.
- Add `docs-only` only when the PR is documentation-only.

**Issue-specific notes:**
- New issues start with `needs-triage` until a maintainer engages.
- Issue templates auto-apply type + `needs-triage` (see taxonomy).

## Report Output

All reports go to `reports/`:

| Workflow | Output file |
|---|---|
| Issue triage | `reports/triage_<YYYY-MM-DD>.md` |
| PR review | `reports/pr_review_<YYYY-MM-DD>.md` |
| Review memory | `reports/review-memory/<pattern-name>.md` |

## Dispatch

After reading this file and the shared references above, read the appropriate
workflow file and follow its instructions:

- **Issue triage:** [triage-workflow.md](triage-workflow.md)
- **PR label & review:** [pr-review-workflow.md](pr-review-workflow.md)
