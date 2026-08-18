# Evaluation Report

Evaluation of the `nemo-mbridge-perf-vision-context-parallel` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `nemo-mbridge-perf-vision-context-parallel`
- Evaluation date: not yet run
- NVSkills-Eval profile: `external`
- Environment: `local`
- Dataset: 3 evaluation tasks
- Attempts per task: 2
- Pass threshold: 50%
- Overall verdict: **PENDING — NVSkills-Eval has not been run against this skill**

> This file is a placeholder committed alongside the skill. The result tables
> below are intentionally empty. Populate them by running NVSkills-Eval with the
> `external` profile against `evals/evals.json`, then replace this notice.
> Do not hand-author scores.

## Agents Used

- `claude-code`
- `codex`

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security` (Security): checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency` (Efficiency): checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy` (Accuracy): grades final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): checks whether the overall user task completed successfully.
- `behavior_check` (Behavior Check): verifies expected behavior steps, including safety expectations.
- `token_efficiency` (Token Efficiency): compares token usage with and without the skill.

## Test Tasks

The benchmark dataset contains 3 evaluation tasks:

- Positive tasks: 2 tasks where the skill was expected to activate — enablement and the load-imbalance diagnosis.
- Negative tasks: 1 task where no skill was expected — a hierarchical context parallelism question for a text-only model, included because "context parallel" is a shared keyword and the two skills describe unrelated mechanisms.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | — | — | — |
| Correctness | — | — | — |
| Discoverability | — | — | — |
| Effectiveness | — | — | — |
| Efficiency | — | — | — |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Not yet run.

## Tier 2: Deduplication Summary

Not yet run. Deduplication is the tier to watch for this skill: it shares the
"context parallel" vocabulary with `nemo-mbridge-perf-hierarchical-context-parallel`
while describing an unrelated mechanism, which is why the dataset includes an
explicit negative routing task.

## Publication Recommendation

Withhold from NVSkills-Eval publication until the evaluation above has been run
and this file updated with real results. The skill content itself is usable in
the repository in the meantime.
