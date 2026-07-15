# Cherrypick Policy

This policy governs what may land on release branches of
**[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** and
**[NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)**
between the cut of a release branch and the corresponding release.

It supplements the cadence and stabilization rules in
[Release Process](release-process.md); read that document first for the
overall release timeline.

-----

## Scope

This policy is in effect from the moment a release branch is cut (start
of code-freeze) until release day. Outside of that window, the normal
contribution rules in [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) apply.

It applies identically to both repositories above. "Cherrypick" here
means any commit landing on a release branch, regardless of whether it
is technically applied via `git cherry-pick`, a backport PR, or a direct
merge into the release branch.

-----

## Two Phases

The release-branch lifetime is split into two phases with different
rules.

| Phase | Window | Rule |
|-------|--------|------|
| **Phase 1 — Code-freeze** | 10 business days; begins on a Monday and ends on the Friday of the following week | Fixes only on in-scope code; out-of-scope changes still allowed |
| **Phase 2 — Absolute freeze** | End of code-freeze through release day | Only changes recommended by QA or the automation team |

In **both phases**, a **single authority is responsible for waiving
cherrypicks onto the release branches**. **This authority is currently
held by the automation team**, but the role is intentionally separable
from any specific team — if ownership moves in the future, the rule
itself does not change. No other party — not authors, not code
reviewers, not QA — can waive a cherrypick on their own. Code review,
blast-radius assessment, and QA recommendations are inputs to the
waiving authority's decision, never substitutes for it.

-----

## Scope of the Freeze Rule

The "fixes only" rule in Phase 1 applies to a defined surface area.
Changes outside that surface area follow the more permissive Phase 1
rules.

### In scope (fixes only during Phase 1)

Source code, tutorials, and examples in either repository:

- **`NVIDIA/Megatron-LM`** — `megatron/` (package source), `examples/`
- **`NVIDIA-NeMo/Megatron-Bridge`** — `src/megatron/bridge/` (package source), `examples/`, `tutorials/`

For these paths, only **fixes** may be cherrypicked during Phase 1.
Feature work does not land on release branches.

### Out of scope (any-allowed during Phase 1)

- **Performance scripts** — e.g. `scripts/performance/` in Megatron-Bridge, and any equivalent benchmarking-only directories in Megatron-LM.
- **Documentation** — `docs/` in either repo, plus top-level Markdown (`README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`, etc.).

Out-of-scope commits may continue to merge throughout Phase 1. They stop
being acceptable as soon as Phase 2 begins — at that point everything,
including docs and perf scripts, falls under the absolute-freeze rule.

-----

## Phase 1 — Code-Freeze

Phase 1 lasts 10 business days: it begins on a **Monday** (the
release-branch cut) and ends on the **Friday of the following week**.

### Blast-radius assessment

Every fix proposed for cherrypick during Phase 1 must have its **blast
radius assessed before merge**. The PR description should explicitly
state:

- What the fix changes.
- Which call sites, models, recipes, or tests can be affected.
- What testing was performed to verify the fix is contained.

Any indication that a fix may be a **breaking change** must be tested
**ahead of merge**, not after. **The waiving authority (currently the
automation team) runs this testing** — verification of breaking-change
risk is part of the waiving authority's role, not the author's. A
confirmed breaking change leads to **rejection of the cherrypick** —
it does not land on the release branch, regardless of how the
underlying issue is later handled on `main`.

### Out-of-scope merges during Phase 1

Performance scripts and documentation may continue to land normally
during Phase 1.

-----

## Phase 2 — Absolute Freeze

Phase 2 starts the moment Phase 1 ends and runs until release day.

During Phase 2, the only acceptable changes are those **recommended by
QA or the automation team**. Examples include:

- A regression QA discovered during release validation.
- A test, golden-value, or annotation fix the automation team requires
  to clear CI for the release.

To raise such a change:

1. Open a PR against the release branch.
2. Tag the relevant QA contact and `@nvidia-nemo/automation`.
3. Link the QA finding or automation ticket that justifies the change.

A QA recommendation is what *initiates* a Phase 2 cherrypick. It is not
itself a waiver — only the waiving authority (currently the automation
team) can waive a cherrypick onto the release branch.

-----

## Quick Reference

| Window | Source code (MLM + MBridge, incl. tutorials/examples) | Performance scripts | Documentation |
|--------|------------------------------------------------------|---------------------|---------------|
| **Phase 1 — Code-freeze** | Fixes only; blast-radius assessed; confirmed breaking changes rejected | Allowed | Allowed |
| **Phase 2 — Absolute freeze** | QA / automation-recommended only | QA / automation-recommended only | QA / automation-recommended only |

In every cell above, the waiving authority — currently the automation
team — is the sole waiver of the cherrypick onto the release branch.

-----

## See Also

- [Release Process](release-process.md) — full release cadence, RC schedule, and golden-values policy.
- [Releases overview](README.md) — index of all release-related documentation.
