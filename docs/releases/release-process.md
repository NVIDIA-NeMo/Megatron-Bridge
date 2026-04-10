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

Golden values are reference outputs used to validate model behavior in CI.

### During the RC Phase (before code-freeze)

Golden values are updated **selectively**:

- They are updated if the new values represent an **improvement**, or
- If the team **collectively decides** that a regression is acceptable.

This means golden values are not automatically updated with every run — a deliberate decision is required for any regression.

### On the Release Branch (during code-freeze)

When the release branch is created at code-freeze, all golden values are updated **unconditionally**. Whatever the current output is becomes the new reference baseline for the release.

-----

## Code-Freeze

Code-freeze lasts **two weeks** and begins when RC3 is cut. This is the **stabilization phase** — no new features are landed.

### First Half

- **Release branches are created.**
- All golden values on the release branch are updated unconditionally (see above).
- The **last bulk CI run** occurs one week into the code-freeze period.
- RCs continue to be cut as needed.

### Second Half

- **Engineers are responsible for updating golden values** on the release branch — reviewing any remaining discrepancies and ensuring the suite is in a clean state ahead of release.
- RCs continue to be cut as needed.

### Release Day

The release goes out on the **first Wednesday after the code-freeze window ends**.

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
