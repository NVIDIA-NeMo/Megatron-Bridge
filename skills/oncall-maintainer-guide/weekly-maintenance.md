# Weekly Maintenance Workflow

## Schedule

| Day | Focus |
|---|---|
| **Monday** | PR label + brief review |
| **Tuesday** | Issue label + triage |
| **Thursday** | PR label + brief review |
| **Friday** | Issue label + triage |

Wednesday is buffer — use for follow-up items generated earlier in the week.

---

## Part A: Follow-Up Check (Every Run)

Run this section on every invocation regardless of day.

### A1: Check `needs-follow-up` items

For each issue/PR with the `needs-follow-up` label, determine whether the
author has responded since the label was applied.

### A2: Check `needs-author` items

For each issue/PR with the `needs-author` label, check whether the author
has responded or pushed changes.

### A3: Classify each item

- **Updated** — author responded or pushed; needs maintainer re-review.
- **Stale** — no author activity; note days since last update.
- **Resolved** — the issue/PR was addressed (merged, closed, or fix landed).
  Recommend removing the label.

---

## Part B: PR Label + Review (Monday & Thursday)

### B1: Collect open PRs

Gather all open, non-draft PRs.

### B2: Label each PR

Apply labels per the label taxonomy:

- One `type` label (bug, feature, chore, docs, ci, etc.)
- One `area` label (area:model, area:recipe, area:training, etc.)
- At most one state label
- Check author affiliation for `community-request`

### B3: Brief review

Focus only on critical and major findings:

- Correctness bugs, behavioral regressions, compatibility breaks
- Unsafe defaults, missing tests for risky new logic
- Merge-readiness blockers (conflicts, dirty branches)

### B4: Priority assignment

| Priority | Criteria |
|---|---|
| **P0 — Urgent** | Blocking other work, CI fix, security issue, or `ready-to-merge` with green CI |
| **P1 — High** | Bug fix, external contributor waiting > 3 days, approved but not merged |
| **P2 — Normal** | Feature PRs with active review, routine updates |
| **P3 — Low** | Draft PRs, docs-only, low-risk chores |

---

## Part C: Issue Label + Triage (Tuesday & Friday)

### C1: Fetch `needs-triage` issues

Collect all open issues with the `needs-triage` label.

### C2: Classify each issue

1. Read the full body and comments.
2. Apply one `type` label, one `area` label.
3. Check author affiliation — note `community-request` if external.
4. Check for existing PRs from the author or linked PRs.
5. Categorize actionability: known fix, needs more info, or other response.

### C3: Priority assignment

| Priority | Criteria |
|---|---|
| **P0 — Urgent** | Data loss, crash on supported path, security vulnerability |
| **P1 — High** | Bug affecting multiple users, regression, external contributor bug with repro |
| **P2 — Normal** | Feature requests with clear use case, support questions, single-user bugs |
| **P3 — Low** | Enhancement ideas, cosmetic issues, questions already answered in docs |

### C4: Prepare triage actions

For each issue, determine the proposed action:

- **Known fix** — describe the fix, plan a PR
- **Needs info** — draft a question asking the author for specifics
- **Other** — draft a guidance response, redirect, or acknowledgment

---

## Rules

- **Creation order**: List items oldest first within each priority tier.
- **Priority first**: Group by priority (P0 > P1 > P2 > P3).
- **Label before review**: Always apply labels before doing brief review or
  triage assessment.
- **Do not post without approval**: The report is the deliverable — the
  maintainer decides what to execute.
- **Concise findings**: For PR reviews, only report critical and major
  findings. Skip style nits.
- **Follow-up is signal**: The follow-up tracker is the most important
  section for continuity across the week. Treat updated items as higher
  priority than new items of the same severity.
