## Description: <br>
Operational guide for sharding a VLM vision encoder across the language model's context-parallel ranks in Megatron-Bridge, including config knobs, code anchors, load-balance pitfalls, and measured impact. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training vision-language models in Megatron-Bridge whose vision tower replicates work and activations on every context-parallel rank while the language model is already sharded, or who need to diagnose why enabling that sharding produced a smaller-than-expected memory saving. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

Risk: The flag silently self-disables at `context_parallel_size=1`, so guidance could be applied to a configuration where it has no effect. <br>
Mitigation: The skill states the CP>1 requirement in the enablement snippet, the pitfalls list, and the verification success criteria. <br>

## Reference(s): <br>
- [Megatron Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) <br>
- [Nemotron-3 Omni example README](../../examples/models/nemotron/nemotron_3_omni/README.md) <br>
- [card.yaml](card.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash and python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Dataset prepared with 3 evaluation tasks (2 positive skill-activation cases, 1 negative routing case) for the NVSkills-Eval external profile. The evaluation has not yet been run. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
Pending — NVSkills-Eval has not been run against this skill. See [BENCHMARK.md](BENCHMARK.md). Do not hand-author scores into this table. <br>

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | — | — | — |
| Correctness | — | — | — |
| Discoverability | — | — | — |
| Effectiveness | — | — | — |
| Efficiency | — | — | — |

## Testing Completed: <br>
**[ ] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
Unreleased — pending first NVSkills-Eval run and signing. <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
