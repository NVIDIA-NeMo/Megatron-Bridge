# Release Developer Guide

## Overview

Our release cycle spans **2 months**. During this window, we develop and land features through a series of Release Candidates (RCs), before entering a code-freeze period for stabilization and a final release.

-----

## Release Candidate Cadence

New RCs are cut every **Saturday**, when the weekly pipeline runs.

|RC |Approximate Timing|Key Activity                      |
|---|------------------|----------------------------------|
|RC0|Week 1 (7th–10th) |Major dependency bump: NGC PyTorch|
|RC1|Week 2            |Dependency bump: TransformerEngine|
|RC2|Week 3            |Feature development continues     |
|RC3|Week 4            |**Code-freeze begins**            |
|   |Week 5            |Bug fixes, small improvements     |
|   |Week 6            |Bug fixes, small improvements     |
|   |Week 7            |QA exit, release                  |

RC0 through RC2 are a **feature development phase** — new features are actively being landed. Stabilization begins at RC3 with code-freeze.

From RC3 onward, RCs are cut **more frequently and as needed**, rather than strictly on Saturdays.

-----

## Golden Values

Golden values are reference outputs used to validate model behavior in CI. They live in the **internal CI repository** and are the baseline for the internal regression tracker — keeping them current and accurate is therefore critical for meaningful signal.

### When to update golden values

Any PR that can affect performance metrics (e.g. changes to model code, training loop, optimizer, or numerical kernels) **must be accompanied by a corresponding internal PR that updates the golden values** before merging. Do not wait until after the PR lands.

### Updating golden values for PRs targeting `main`

1. **Rebase the MBridge PR against `main`** so it is at top-of-tree before launching CI.
2. **Launch an internal CI run** using:
   - The **latest nightly container** as the base image.
   - The **latest MCore commit** on `main`.
   - The **MBridge PR commit** (the head of your MBridge branch).
3. Collect the outputs and open a PR against the **internal CI repository's `main` branch** with the updated golden values.
4. The MBridge PR and the internal golden-values PR should be merged together (or the golden-values PR first).

### Updating golden values during a release

When golden values need to be refreshed on the release branch (e.g. at the start of code-freeze or after an accepted regression):

1. **Rebase the MBridge PR against the MBridge release branch** so it is at the head of that branch.
2. **Launch an internal CI run** using:
   - The **latest internal RC container** for the release.
   - The **MCore commit pinned on the release branch**.
   - The **MBridge PR commit** (head of the MBridge release branch).
3. Open a PR against the **internal CI repository's release branch** with the updated golden values.

### During the RC Phase (before code-freeze)

Golden values are updated **selectively**:

- They are updated if the new values represent an **improvement**, or
- If the team **collectively decides** that a regression is acceptable.

This means golden values are not automatically updated with every run — a deliberate decision is required for any regression.

### On the Release Branch (during code-freeze)

When the release branch is created at code-freeze, all golden values are updated **unconditionally** — whatever the current output is becomes the new reference baseline for the release.

In **Week 5**, the last bulk update of golden values is performed. After that point, engineers are individually responsible for updating any remaining golden values on the release branch, reviewing discrepancies and ensuring the suite is clean ahead of the release.

-----

## Code-Freeze

Code-freeze lasts **two weeks** and begins when RC3 is cut. This is the **stabilization phase** — no new features are landed.

### First Half (Weeks 3–5)

- **Release branches are created.**
- All golden values on the release branch are updated unconditionally (see above).
- The **last bulk update of golden values** happens in **Week 5**.
- RCs continue to be cut as needed.

### Second Half (Weeks 6–7)

- **Engineers are individually responsible for updating golden values** on the release branch — reviewing any remaining discrepancies and ensuring the suite is in a clean state ahead of release.
- RCs continue to be cut as needed.

### Release Day

The release goes out on the **first Wednesday after the code-freeze window ends**.

-----

## Patch Release

After the main release ships (Week 7, typically mid-month), the release branches are **reopened** for contributions targeting the patch release.

### Patch Release Timeline

| Period | Approximate Timing | Key Activity |
|--------|--------------------|--------------|
| Reopening | Release day (Week 7) | Branches accept contributions; patch development begins |
| Lockdown | First Monday of the following month (~2 weeks later) | Release branches locked; **patch RC0 (`XX.YY.01.RC0`) shipped internally** |
| Stabilization | Week 1–2 after RC0 | Bug fixes and small improvements only |
| | End of Week 2 | QA exit, patch release |

The patch stabilization flow mirrors the main release's code-freeze phase, but compressed into approximately two weeks.

-----

## CI and Known Failures

### Ticket-Annotated Tests

Failing CI tests can be linked to a tracking ticket. When a test fails with the **same error code** as the one recorded on its linked ticket, CI reports it as **"passing, with known error"** rather than a hard failure.

This means **a green CI result does not guarantee a fully healthy test suite** — it means there are no *unexpected* failures.

### Important: Keeping Annotations Up to Date

Ticket annotations must be actively maintained in **both directions**:

- **Add** a ticket annotation when a test starts failing with a known, accepted error.
- **Remove** the ticket annotation when the test heals.

If a test recovers but its ticket annotation is not removed, CI will report it as **failing** — because the actual error code no longer matches the one on record. The test being healthy is not enough; the annotation must be cleaned up for CI to go green again.
