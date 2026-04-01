# Triage Report Template

Use this exact structure when generating `reports/triage_<YYYY-MM-DD>.md`.

---

```markdown
# Triage Summary — <DATE>

<N> issues with `needs-triage`. <M> have 0 comments and 0 assignees.

**Strategy:** For issues with confirmed, actionable fixes — open a PR, share it in the response, and move to `needs-follow-up`. For issues needing more info — ask the author and add `needs-author`.

---

## Issues with Known Fixes (will create PRs)

---

### #<NUMBER> — <TITLE>

| Field | Value |
|---|---|
| **Type** | `<type>` (<already set / inferred>) |
| **Area** | `<area>` (<already set / changed from X>) |
| **State** | `needs-triage` → `<new-state>` |
| **Author** | <username> (<internal / external, note community-request if applicable>) |
| **Label changes** | remove `needs-triage`, add `<new-state>` |
| **Link** | https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/<NUMBER> |

**Fix:** <Exact description: file path, line numbers, what to change, with code diff>

```python
# Before:
<old code>
# After:
<new code>
```

**Proposed response:**

> <Response text. Use blockquote format. Include `<PR_LINK>` placeholder for PR URL.>

---

(repeat for each known-fix issue)

---

## Issues Needing More Info (no PR, ask author)

---

### #<NUMBER> — <TITLE>

| Field | Value |
|---|---|
| **Type** | `<type>` |
| **Area** | `<area>` |
| **State** | keep `needs-triage`, add `needs-author` |
| **Author** | <username> |
| **Label changes** | add `needs-author` |
| **Link** | https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/<NUMBER> |

**Assessment:** <Why we can't fix yet — what's missing>

**Proposed response:**

> <Response asking for details: repro script, env info, error logs, etc.>

---

(repeat for each needs-info issue)

---

## Issues with Other Responses (no PR needed)

---

### #<NUMBER> — <TITLE>

| Field | Value |
|---|---|
| **Type** | `<type>` |
| **Area** | `<area>` |
| **State** | `needs-triage` → `<new-state>` |
| **Author** | <username> |
| **Label changes** | <changes> |
| **Link** | https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/<NUMBER> |

**Proposed response:**

> <Answer, redirect, or guidance>

---

(repeat for each other-response issue)

---

## Overview

| # | Title | Type | Area | New State | Action |
|---|---|---|---|---|---|
| <NUMBER> | <short title> | <type> | <area> | <new-state> | **PR** / Ask for details / Answer directly |

### PR Plan (<N> PRs)

| PR Branch | Issues | Files Changed |
|---|---|---|
| `yuya/<branch-name>` | #<NUMBER> | `<relative path from src/megatron/bridge/>` |

### Patterns

- <Notable patterns: frequent reporters, related issues, combined PRs, etc.>
```

---

## Notes

- **`<PR_LINK>` placeholders** get replaced with actual PR URLs during Phase 3 execution.
- **SKIP/DONT RESPOND** — if the maintainer writes this on an issue, skip responding and relabeling for that issue.
- **Combined PRs** — when multiple issues share a root cause, plan a single PR and note "combined w/ #X" in the overview.
- Every issue entry must include a `**Link**` field with the GitHub URL so the maintainer can click through during review.
