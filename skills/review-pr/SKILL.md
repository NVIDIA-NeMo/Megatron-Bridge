---
name: review-pr
description: Structured single-agent code review workflow for PRs, commits, and local diffs. Use when asked to review code, understand a PR, rubber duck a change, prepare GitHub review comments, compare a change against Megatron Bridge conventions, or produce high-signal findings without subagents or tmux.
when_to_use: Reviewing a GitHub PR, commit, local diff, or code change; preparing review comments; checking a change against Megatron Bridge conventions; assessing whether a PR is safe to merge; summarizing review findings and test gaps.
---

# Review PR

Use this skill to review a change in staged passes. Do not use subagents or tmux unless the user explicitly asks for them.

## Ground Rules

- Pull the artifact first: read the PR, commit, diff, log, or files under review before forming conclusions.
- After the artifact is read, load only the relevant repo skills for the touched areas, such as `linting-and-formatting`, `testing`, `adding-model-support`, `build-and-dependency`, `cicd`, or a performance skill.
- Preserve user changes. Do not edit files during a review unless the user asks for fixes.
- Do not modify `3rdparty/Megatron-LM/`. If the reviewed change touches it, flag that boundary.
- Do not run the full test suite. Run only targeted checks, using `uv run python -m pytest` or the repo-approved command form.
- Prefer findings with concrete file and line evidence. Drop low-confidence or purely stylistic comments unless the user asks for a strict style pass.

## Repository Review Principles

Apply these principles to implementation, APIs, recipes, tests, examples, and
documentation. They are acceptance criteria, not optional polish.

1. **Correctness is the gate.** Preserve numerical semantics, distributed
   invariants, checkpoint compatibility, API contracts, and failure behavior.
   Prefer an explicit error over a silent fallback that can produce incorrect
   training, conversion, or inference results. Performance or simplicity never
   justifies behavior that is wrong or cannot be validated.
2. **Performance is a product requirement.** Once correctness is established,
   protect throughput, latency, memory efficiency, and distributed scaling,
   especially in training hot paths. Do not accept an avoidable regression for
   cleaner-looking code or a more convenient abstraction without measurements
   and an explicit tradeoff. Treat extra synchronization, communication,
   materialization, copies, allocations, and host overhead as review concerns.
3. **Design for both users and developers.** User-facing behavior should have
   safe defaults, coherent configuration, actionable errors, and discoverable
   examples. Developer-facing code should have explicit contracts, clear names
   and ownership, local reasoning, focused tests, and minimal special cases.
   Readability includes the public workflow as well as the implementation.
4. **Keep optimized complexity behind clear boundaries.** A fast implementation
   may be internally specialized, but it should not leak accidental complexity
   into public APIs or every call site. Isolate the optimized path, document its
   invariants, and test both its behavior and fallback or error path.

Use this decision order:

- Block incorrect or silently ambiguous behavior, even if it is faster.
- Block an unjustified performance regression in a hot or distributed path;
  require comparable measurements when the impact is material.
- Between solutions with comparable correctness and performance, prefer the one
  that is easier for users to operate and developers to understand and extend.
- When usability and performance genuinely conflict, require the PR to state
  the tradeoff and keep the simpler contract while containing optimized details.

## Wave 0: Intake

Identify the review target and collect enough raw context to avoid guessing.

For a GitHub PR:

```bash
gh pr view <PR> --json number,title,state,baseRefName,headRefName,mergeStateStatus,author,files,comments,reviews
gh api repos/<owner>/<repo>/pulls/<PR>/comments --paginate
gh pr diff <PR> --name-only
gh pr diff <PR>
```

For a local diff or commit, inspect status, stat, and patch:

```bash
git status --short
git diff --stat
git diff
git show --stat <commit>
git show <commit>
```

Then inspect surrounding code, call sites, configs, tests, and imports with `rg`, `sed`, `git show`, or targeted file reads. Record the changed files, changed APIs, affected workflows, and any existing comments or failing logs.

## Wave 1: Scope And Intent

Build a short private map before judging the code:

- What behavior is being added, removed, or changed?
- Which contracts are touched: public APIs, dataclass configs, checkpoints, recipes, conversion mappings, distributed launch behavior, dependency pins, CI, or docs?
- Which users or downstream systems are affected: HF import/export, MCore training, NeMo-RL, verl, functional tests, CI, or performance recipes?
- Which relevant skills should be loaded now based on the touched files?

If the requested output is PR understanding rather than review comments, keep this wave and output a concise architecture/change summary after Wave 3.

## Wave 2: Specialist Lenses

Run these passes sequentially as one agent. Keep notes separate so one lens does not bias the next.

### Correctness And Contracts

Check for logic bugs, shape or dtype mismatches, device movement mistakes, serialization compatibility, broken invariants, missing error handling, API misuse, and changes that silently alter existing behavior.

For model or conversion changes, verify HF/MCore name mappings, tensor transforms, config mapping defaults, tied weights, sharding assumptions, AutoBridge selection, and parity-test implications.

For changes that appear specific to one model or one training feature, check whether neighboring models or features share the same path and could regress. If common infrastructure is modified, examine it carefully and require the change to be necessary, generalizable, and covered by tests or clear reasoning.

### Integration And Runtime Behavior

Trace how the change is reached in real workflows. Follow call sites across recipes, providers, launch scripts, configs, and downstream adapters. For distributed or performance-sensitive code, check rank behavior, process-group assumptions, collective ordering, recompute/offload/overlap interactions, and GPU-count constraints.

### User And Developer Experience

Review the complete user workflow, not only the changed function. Check whether
defaults, configuration names, errors, docs, examples, and migration behavior
make the feature safe and discoverable. Then review the developer workflow:
whether contracts and ownership are explicit, names match semantics, the control
flow can be understood locally, and new models or features can reuse the path
without copying model-specific glue.

Do not approve an abstraction solely because it reduces lines of code. It must
also reduce user or developer complexity without obscuring runtime behavior.

### Performance And Scalability

Identify whether the change touches a hot path or changes communication,
parallelism, memory lifetime, kernel selection, graph capture, data movement, or
checkpoint I/O. Check end-to-end step time and peak memory where relevant, not
only isolated kernel speed. Include startup, compile, capture, and scaling costs
when they can dominate the actual workflow.

For a claimed optimization or a material performance-sensitive change, ask for
an apples-to-apples baseline using the same model, hardware class, parallelism,
sequence length, and batch shape. If measurement is not feasible, require a
bounded explanation of the expected cost and explicitly record the residual
risk. Do not infer performance from code shape alone.

### Tests And Verification

Check whether tests cover the changed contract, failure mode, and representative runtime path. Prefer unit tests unless the risk requires a functional test. In functional tests, enforce the repo rule that override patterns using `setattr(config_obj, key, value)` must first guard with `hasattr`.

Run targeted checks only when useful and feasible. If a check cannot be run, state why.

### Style, API, And Maintainability

Check public type hints, Google-style docstrings for public APIs, logging instead of bare `print()`, explicit config values, keyword-only separators for ambiguous same-type parameters, dependency policy, and CI/public-API boundaries. Do not over-weight style issues that tooling will catch unless they hide a real review-worthy problem.

## Wave 3: Adversarial Verification

Challenge every candidate finding before showing it to the user.

For each candidate:

- Try to disprove it by reading the surrounding implementation and call sites.
- Confirm the affected line is reviewable, preferably an added or modified diff line.
- Identify the exact failure scenario, user impact, or broken repository rule.
- Check whether existing tests already cover it.
- Check it against the repository principles: correctness, measured or reasoned
  performance impact, and user/developer clarity.
- Merge duplicates that point to the same root cause.
- Assign a verdict: `CONFIRMED`, `DOWNGRADED`, `QUESTION`, or `DROP`.

Only keep inline-review findings that are actionable and high confidence. As a default threshold, keep findings at confidence `>=80`; turn lower-confidence concerns into open questions or omit them.

## Wave 4: Assemble The Review

Lead with findings, ordered by severity. Use this shape:

```text
- [Severity] Title
  Location: path/to/file.py:123
  Problem: What is wrong.
  Evidence: Why the code proves it.
  Impact: What can break or why it matters.
  Suggested fix: Concrete remediation.
  Confidence: 90
```

Use clickable file links when the response environment supports them. Keep summaries brief and after the findings. Include:

- Open questions or assumptions.
- Tests or commands run, with pass/fail status.
- Tests not run and why.
- A short change summary only as supporting context.

If there are no findings, say so directly and name the remaining test gaps or residual risks.

## GitHub Review Protocol

When the user asks to prepare or submit a GitHub review:

1. Present the proposed inline comments first.
2. Ask for explicit confirmation before publishing comments.
3. Keep the review body short; actionable issues belong inline.
4. Do not publish noisy, speculative, or pedantic comments. A shorter review with confirmed issues is preferred.

If pending-review tooling is available, stage a pending review for preview before publishing. Otherwise, provide the exact comment set and wait for confirmation before using `gh` or the GitHub API.

## Severity Guide

- `Critical`: Data corruption, incorrect checkpoint conversion, deadlock, security issue, or guaranteed runtime failure in a core path.
- `High`: Real bug with likely user impact, broken public contract, missing required test for risky behavior, or CI-breaking issue.
- `Medium`: Plausible bug or maintainability issue with narrower impact and clear remediation.
- `Low`: Minor issue worth fixing but not merge-blocking.
- `Question`: Something that needs author clarification before becoming a finding.
