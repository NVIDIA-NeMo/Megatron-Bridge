# PR Review Report — <YYYY-MM-DD>

<N> PRs reviewed. <M> findings above threshold (score >= 80). <K> low-confidence findings filtered.
Review mode: brief | deep.

---

## High-Confidence Findings

List rows newest to oldest by PR number / recency.

| Score | Category | PR | Finding |
| --- | --- | --- | --- |
| 95 | BUG | [#NNNN](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/NNNN) | Short description with file:line reference. |
| 85 | REGRESSION | [#NNNN](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/NNNN) | Short description. |

If there are no findings above the 80-score threshold, say:

```markdown
No findings above the 80-score threshold across the reviewed PRs.
```

---

## Per-PR Review

Order PR subsections newest to oldest by PR number / recency.

### #NNNN — PR title

| Field | Value |
| --- | --- |
| Author | `<login>` |
| Area | `area:*` |
| Labels | `bug`, `area:model`, `needs-review` |
| CI / Merge | `summary` |

Summary: One or two sentences on what changed and why it matters.

Findings (scored >= 80):

- [BUG 95] `file.py:42` — Concrete issue description.
- [TEST 82] `test_file.py` — Missing coverage for new code path.

Filtered: N findings below threshold omitted.

If none:

```markdown
Findings: No findings above threshold.
```

---

## Optional Follow-Up

- PRs to request changes on: `#...`
- PRs safe to approve from a major-risk perspective: `#...`

---

## Review Memory Updates

Patterns confirmed during this review that were added to `reports/review-memory/`:

- `<pattern-name>`: first seen | seen N times (promote to code-style skill?)

If none:

```markdown
No new review-memory patterns.
```
