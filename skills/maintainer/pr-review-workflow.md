# PR Label & Review Workflow

Maintainer workflow for labeling and reviewing pull requests. Supports brief
(default) and deep review modes.

Prerequisites: read [SKILL.md](SKILL.md) first for shared setup, label taxonomy, and author checks.

## Parse Arguments

- "deep review" or "--deep" → **deep mode** (parallel subagents)
- Otherwise → **brief mode** (single-agent review)
- "post comments" or "submit review" → post findings as GitHub review comments
- Otherwise → report only (no GitHub comments unless explicitly asked)

If the user asks for only labeling or only review, do only that part.

---

## Collect PRs

If the user gives PR numbers, use those.

Otherwise collect PRs with `gh pr list`, for example:

```bash
gh pr list --repo NVIDIA-NeMo/Megatron-Bridge --state open --limit 100 \
  --json number,title,author,isDraft,labels,url
```

For `needs-review` sweeps:

```bash
gh pr list --repo NVIDIA-NeMo/Megatron-Bridge --state open --label needs-review \
  --limit 100 --json number,title,author,isDraft,labels,url
```

For each PR you touch, fetch metadata first:

```bash
gh pr view <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge \
  --json number,title,author,isDraft,labels,reviewDecision,reviewRequests,mergeStateStatus,state,files,statusCheckRollup
```

---

## Labeling

Apply labels using the shared rules in [SKILL.md](SKILL.md). Additionally:

### State label heuristics (PR-specific)

Use at most one state label.

- Draft PR: no state label
- `ready-to-merge`: approved, current, and checks are green
- `needs-author`: merge conflicts, dirty branch, requested changes, or obvious author follow-up needed
- `needs-review`: non-draft PR waiting on review
- `blocked`: only when there is a real external blocker, not just GitHub's generic `BLOCKED` merge state

Important:
- Do not use `needs-triage` on PRs.
- Do not infer `blocked` from GitHub's `mergeStateStatus: BLOCKED` alone. GitHub often uses that state for ordinary "still waiting on review/checks" conditions.
- If a PR is merely waiting for review, prefer `needs-review`.

### Docs-only label

Add `docs-only` only when the PR is documentation-only.

Remove `docs-only` if the diff touches executable code, workflows, tests, or dependencies.

### Apply labels

Compute the desired labels first, then apply only the needed changes:

```bash
gh pr edit <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge \
  --add-label "<label>" \
  --remove-label "<wrong-label>"
```

When changing the state label, remove any old state label that no longer applies.

---

## Review

### Phase 1: Gather Context

For each PR, fetch the diff and changed file list:

```bash
gh pr diff <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge
gh pr view <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge --json files
```

Then read only the relevant local files for context. Prefer the changed
symbols and adjacent code over whole-file reads.

### Phase 2: Analyze

#### Brief mode (default)

Analyze all changes yourself. Focus on:
- correctness bugs
- behavioral regressions
- compatibility breaks across MCore or config versions
- unsafe defaults or silent fallback behavior
- missing tests for risky new logic
- unrelated scope expansion, such as surprise submodule bumps in a narrow fix
- merge-readiness blockers like conflicts or dirty branches
- violations of coding guidelines from the code-style skill
- patterns from review-memory files

Ignore pure style nits unless they rise to a major issue.

Produce a list of candidate findings, each with: file, line, category,
and description.

#### Deep mode (parallel subagents)

Launch 3 subagents in parallel. Provide each with the diff, PR description,
and the code-style skill content.

**Subagent 1 — Guideline compliance:**
Review the diff against the code-style skill. For
each violation, return the file, line, description, and which rule it violates.

**Subagent 2 — Bug scan (diff only):**
Scan for bugs in the diff without reading surrounding context. Flag type
errors, logic errors, missing imports, unresolved references, shape mismatches
in tensor operations.

**Subagent 3 — Contextual bug scan:**
Read surrounding code for each changed file. Look for bugs that only appear
with context: incorrect MCore API usage, broken conversion mappings, race
conditions in distributed code, wrong TP/PP assumptions.

After all subagents return: merge results and deduplicate (same file + line +
issue = one finding).

### Phase 3: Validate and Score

For each candidate finding, assign a confidence score:

| Score | Meaning |
|---|---|
| 0–25 | Low confidence, likely false positive |
| 26–50 | Moderate, might be real but minor |
| 51–75 | Confident, real and worth noting |
| 76–89 | High confidence, real and important |
| 90–100 | Certain, must-fix |

**Filter:** discard findings scoring below **80**. These are likely false
positives or insignificant nits.

Categorize surviving findings:
- **[BUG]** — Logic errors, shape mismatches, null refs, race conditions
- **[REGRESSION]** — Behavioral changes that break existing functionality
- **[COMPAT]** — MCore or config version compatibility issues
- **[TEST]** — Missing or insufficient test coverage for risky new logic
- **[GUIDELINE]** — Violations of coding guidelines (only if significant)

### Phase 4: Report

Display findings to the user with confidence scores:

```
PR #<number>: <title> (by <author>)
Files changed: <count>

--- Findings (scored >=80) ---
[BUG 95] path/to/file.py:42 — Brief description
[REGRESSION 85] path/to/other.py:15 — Brief description

--- Filtered (scored <80) ---
N low-confidence issues omitted
```

If a large batch is requested, review in parallel with subagents. A good
default is batches of 5-8 PRs.

Write the report to `reports/pr_review_<YYYY-MM-DD>.md` using the template
in [pr-review-report-template.md](pr-review-report-template.md). If a report
already exists for the same date, update it.

### Phase 5: Post Review (only if user explicitly asks)

If the user asks to "post comments" or "submit review":

```bash
gh pr review <NUMBER> --repo NVIDIA-NeMo/Megatron-Bridge \
  --comment --body "<review body>"
```

Or for line-specific comments, use the GitHub API:

```bash
gh api repos/NVIDIA-NeMo/Megatron-Bridge/pulls/<NUMBER>/reviews \
  --method POST \
  -f event=COMMENT \
  -f body="<summary>" \
  -f 'comments[][path]=<file>' \
  -f 'comments[][position]=<line>' \
  -f 'comments[][body]=<comment>'
```

### Phase 6: Update Review Memory

After the review, for each finding the user confirmed or discussed:

1. Check `reports/review-memory/*.md` for existing patterns.
2. If a matching memory file exists, add the new occurrence to its
   `## Occurrences` section. If the pattern has appeared 3+ times, suggest
   promoting it to the code-style skill.
3. If no matching memory file exists, create a new one:

```markdown
# <Pattern Name>

**Do:** <what to do instead>
**Don't:** <the anti-pattern>

## Occurrences
- PR #<number>: <file>:<line> (<date>)
```

Create the `reports/review-memory/` directory if it does not exist.

---

## Output Rules

Keep the report concise and skimmable:
- findings first, with confidence scores
- order both the summary table and per-PR sections newest to oldest
- one subsection per PR
- explicitly say when a PR has no findings above the 80-score threshold
- include a "Filtered" count so the reader knows how many low-confidence
  items were suppressed

Do not bury the lead in prose. Start with a top summary table or headline
counts.

## Recommended Execution Order

1. Gather PRs.
2. Read shared setup from [SKILL.md](SKILL.md).
3. Read review-memory files if any exist.
4. Tag PRs if the user asked for labeling.
5. Review diffs and risky files (brief or deep mode).
6. Score and filter findings.
7. Write `reports/pr_review_<YYYY-MM-DD>.md`.
8. Update review memory for confirmed findings.
9. Reply with the report path and a short summary of blockers.

## Examples

User asks:
- "help me tag those PRs" → labeling only
- "review all PRs tagged needs-review" → brief review + labeling
- "deep review PR #123" → deep mode with parallel subagents
- "review and post comments on #456" → brief review + post to GitHub
- "do a brief PR sweep" → brief review of all open PRs
- "label and triage open PRs" → labeling + brief review
