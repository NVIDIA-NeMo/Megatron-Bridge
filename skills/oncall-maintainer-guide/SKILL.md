---
name: oncall-maintainer-guide
description: Introduction to Megatron Bridge oncall and maintainer roles, responsibilities, cadence, and workflows. Use when onboarding to oncall, understanding maintainer duties, or looking up who owns what.
---

# Oncall & Maintainer Guide

This guide introduces the two key operational roles for Megatron Bridge:
**Oncall** (rotating) and **Maintainer** (permanent). It covers what each role
owns, the weekly cadence, and how to handle common situations.

## Maintainers

| GitHub handle | Name |
|---|---|
| `yaoyu-33` | Yu Yao |
| `chencuix` | Chen Cui |

Maintainers have merge authority and can override CI requirements when needed
(see [Merge Queues](#merge-queues) below).

---

## Responsibility Matrix

| Area | Cadence | Owner | Label / Task | What to do |
|---|---|---|---|---|
| **Issues** | Ongoing | Oncall | `needs-follow-up` | Respond to issues where authors have replied and are awaiting follow-up |
| | Tue, Fri | Maintainer | `needs-triage` | Classify, label, and assign incoming issues |
| | Ongoing | Author | `needs-author` | Author to clarify question |
| **PRs** | Mon, Thu | Maintainer | `needs-review` | Review open PRs ready for code review |
| | Ongoing | Oncall | `ready-to-merge` | Monitor CI, restart flaky jobs, follow up on failures |
| | Ongoing | Oncall | `needs-follow-up` | Ping authors on PRs blocked on their response |
| **Daily Bump** | Daily, as needed | Maintainer | `needs-follow-up` | Fix daily bump issues with one or more separate PR(s) |
| **Sync** | Wed | Maintainer | — | Downstream new Megatron-LM features into Megatron-Bridge |

---

## PRs

### Daily Bump

The MCore daily bump is primarily handled by the **maintainer** today. When the
automated bump PR fails CI, the maintainer cherry-picks the bump commit into a
fix branch and lands the fix separately.

For the full bump procedure, see the `oncall-assistant` or `mcore-bump` skill.

### Merge Queues

When a `ready-to-merge` PR fails in the merge queue, follow this decision tree:

1. **Re-run** — If the failure is a timeout, NCCL error, or other transient
   infra issue, re-trigger CI.

2. **Rebase on main** — If the error is unrelated to the current PR (e.g., a
   dependency changed on `main`), rebase the PR onto the latest `main` and
   re-run CI.

3. **Re-approve after rebase** — `ready-to-merge` means the PR is already
   approved. If a rebase is needed and the approval is invalidated, the oncaller
   should re-approve since the code hasn't changed.

4. **Fast merge (maintainer override)** — If you're confident the failure is a
   systemic infra issue and don't want to wait for another full CI run, reach
   out to **Yu Yao** (`yaoyu-33`) or **Chen Cui** (`chencuix`) to fast-merge.

5. **Real errors** — If the failure is a genuine test failure caused by the PR,
   reach out to the PR author to fix it.

---

## Issues

### Prioritization

When triaging issues, prioritize in this order:

1. **Something is breaking** — Active regressions or crashes on supported paths
   get immediate attention.

2. **Partner requests** — Issues from GCP, Amazon, and other partners.
   Check labels and issue context to identify these.

3. **Show-stoppers** — Issues with the `show-stopper` label block releases
   or major workflows.

For the full triage procedure, see the `triage-issues` skill.

---

## Sync

Wednesday is sync day. The maintainer checks for new Megatron-LM commits in
`megatron/training/` and downstreams any relevant changes into Bridge.

For the full procedure, see the `training-sync` skill.

---

## Detailed Workflows

These documents describe each workflow in detail:

| Document | Purpose |
|---|---|
| [weekly-maintenance.md](weekly-maintenance.md) | Mon/Thu PR review, Tue/Fri issue triage, follow-up tracking |
| [triage-issues.md](triage-issues.md) | End-to-end issue triage with review gates |
| [training-sync.md](training-sync.md) | Check MCore training changes and produce a sync report |
| [mcore-bump.md](mcore-bump.md) | MCore submodule bump procedure |
