# Issue Triage Workflow

Four-phase workflow with two maintainer review gates:

**Phase 1:** Fetch & Summarize → **REVIEW #1** → **Phase 2:** Execute PRs + Draft Responses → **REVIEW #2** → **Phase 3:** Post Responses + Relabel

Prerequisites: read [SKILL.md](SKILL.md) first for shared setup, label taxonomy, and author checks.

---

## Phase 1: Fetch & Summarize

### 1a: Fetch all `needs-triage` issues

```bash
gh issue list --repo NVIDIA-NeMo/Megatron-Bridge \
  --label "needs-triage" --state open \
  --json number,title,body,labels,author,createdAt,comments \
  --limit 50
```

### 1b: Read each issue

For each issue, fetch the full body and comments:

```bash
gh issue view <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge \
  --json number,title,body,labels,author,comments,assignees
```

### 1c: Classify each issue

Apply labels from the taxonomy in [label-taxonomy.md](label-taxonomy.md).

For each issue determine:
1. **Type** — one type label from the taxonomy (often already set by issue template).
2. **Area** — one area label from the taxonomy.
3. **Author** — run the author affiliation check from [SKILL.md](SKILL.md).
4. **Existing PR check** — ALWAYS check if the author (or anyone) already opened a PR before planning a new one:
   ```bash
   gh pr list --repo NVIDIA-NeMo/Megatron-Bridge --search "<issue number>" --json number,title,state,author,headRefName --limit 5
   ```
   Also check the author's open PRs:
   ```bash
   gh pr list --repo NVIDIA-NeMo/Megatron-Bridge --author <AUTHOR> --state open --json number,title,headRefName --limit 10
   ```
   If an existing PR covers the fix, link it in the report instead of planning a new branch. Prefer reviewing and merging author PRs over creating duplicates.
5. **Actionability** — categorize into one of:
   - **Known fix (existing PR)** — author already opened a PR; review and merge it
   - **Known fix (new PR needed)** — root cause is clear, no existing PR, create one
   - **Needs more info** — need repro script, env details, or clarification from author
   - **Other response** — support question, feature request, or redirect

### 1d: Generate triage report

Write the report to `reports/triage_<YYYY-MM-DD>.md` using the template in [triage-report-template.md](triage-report-template.md).

The report has these sections:

1. **Issues with Known Fixes** — each issue gets:
   - Metadata table (type, area, state transition, author, label changes)
   - **Fix** description with exact file/line and code diff
   - Whether an existing PR covers the fix (link it) or a new PR is needed
   - **Proposed response** as a blockquote (will be posted as GitHub comment)
   - All PR references must use full links: `[#NNNN](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/NNNN)`

2. **Issues Needing More Info** — each issue gets:
   - Metadata table
   - **Assessment** of what's unclear
   - **Proposed response** asking for details

3. **Issues with Other Responses** — support questions, feature requests, etc.

4. **Overview table** — all issues in one table with columns: `#`, `Title`, `Type`, `Area`, `New State`, `Action`

5. **PR Plan** — table of PRs (both new and existing author PRs) with status

### 1e: Present to maintainer

Tell the user:

> Triage report written to `reports/triage_<DATE>.md`. Please review:
> - Issue summaries and proposed actions
> - Fix descriptions and code diffs
> - PR plan (new PRs vs existing author PRs)
>
> Edit any section directly in the file, then say **"go"** when ready.

**STOP HERE. Wait for the maintainer to review, edit, and approve.**

---

## Review Gate #1: Maintainer Reviews Triage Plan

The maintainer reviews `reports/triage_<DATE>.md` and may:
- Edit proposed responses
- Change fix descriptions
- Add/remove issues from the PR plan
- Write "DONT RESPOND" or "SKIP" on issues to exclude
- Adjust which issues get new PRs vs review existing author PRs

When the maintainer says "go", "execute", or "apply", proceed to Phase 2.

---

## Phase 2: Execute PRs + Draft Responses

Re-read the (possibly edited) `reports/triage_<DATE>.md` and execute PRs. Then finalize draft responses with actual PR links.

### 2a: Create PRs for issues needing new PRs

For each issue marked for a new PR (not covered by an existing author PR):

1. **Create branch** from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b yuya/<branch-name-from-pr-plan>
   ```

2. **Apply the fix** described in the report. Use the exact code changes specified.

3. **Commit, push, create PR** following the git-commit-workflow skill:
   - Run pre-commit
   - Commit with `-s` and conventional commit message
   - Push and create PR with title format `[{area}] fix: {description}`
   - Comment `/ok to test <commit_hash>`

4. **Return to `main`** before starting the next fix:
   ```bash
   git checkout main
   ```

### 2b: Finalize responses in triage report

Update `reports/triage_<DATE>.md`:
- Replace all `<PR_LINK>` placeholders with actual PR links
- All PR references must use full URLs: `[#NNNN](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/NNNN)`
- Update the Overview table and PR Plan with created PR numbers and links
- Close any duplicate PRs (if we created a PR but discovered author already has one)

### 2c: Present draft responses to maintainer

Tell the user:

> PRs created. Draft responses finalized in `reports/triage_<DATE>.md`.
> Please review the responses (blockquoted sections) — these will be posted as-is.
>
> Edit any response directly in the file, then say **"reply"** or **"post"** when ready.

**STOP HERE. Wait for the maintainer to review responses before posting.**

---

## Review Gate #2: Maintainer Reviews Responses

The maintainer reviews the finalized responses in `reports/triage_<DATE>.md` and may:
- Edit response wording
- Add "DONT RESPOND" or "SKIP" to specific issues
- Adjust label changes

When the maintainer says "reply", "post", "respond", or "go", proceed to Phase 3.

---

## Phase 3: Post Responses + Relabel

Re-read the (possibly edited) `reports/triage_<DATE>.md` and execute responses and labels.

### 3a: Post responses on issues

For each issue with a proposed response (not marked SKIP/DONT RESPOND):

```bash
gh issue comment <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge \
  --body "$(cat <<'EOF'
<proposed response with PR links filled in>
EOF
)"
```

### 3b: Relabel issues

For each issue, apply the label changes from its metadata table:

```bash
gh issue edit <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge \
  --add-label "<label1>" --add-label "<label2>" \
  --remove-label "needs-triage"
```

Common state transitions:
- **Known fix with PR** → remove `needs-triage`, add `needs-follow-up`
- **Needs more info** → keep or add `needs-author` (may also keep `needs-triage`)
- **Support answered** → remove `needs-triage`, add `needs-follow-up`

### 3c: Summary

After all actions are complete, print a summary:

```
## Triage Execution Complete — <DATE>

### PRs Created
| # | Issue | PR | Branch |
|---|---|---|---|

### Responses Posted
| # | Title | State |
|---|---|---|

### Skipped
| # | Title | Reason |
|---|---|---|
```

---

## Decision Tree

```
NEEDS-TRIAGE ISSUE
├── Read issue body + comments
├── Classify: type + area
├── Check for existing author PRs (ALWAYS do this)
├── Determine actionability:
│   ├── Known fix (author PR exists) → link existing PR, plan review
│   ├── Known fix (no PR) → plan new PR
│   ├── Needs info → draft question for author
│   └── Other → draft response (support, feature, redirect)
├── Write to triage report
└── REVIEW GATE #1: maintainer approves plan
    └── On "go":
        ├── Create new PRs (known fixes without author PR)
        ├── Finalize draft responses with PR links
        └── REVIEW GATE #2: maintainer approves responses
            └── On "reply":
                ├── Post responses (all non-skipped)
                └── Relabel issues
```
