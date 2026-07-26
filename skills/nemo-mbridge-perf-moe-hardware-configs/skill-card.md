## Description: <br>
Representative MoE training playbooks by hardware platform and model family, summarizing rounded throughput bands, parallelism patterns, and common tuning stacks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and performance engineers selecting MoE training configurations, dispatcher choices, and parallelism layouts for specific hardware platforms (H100, B200, GB200, GB300) and model families (DSV3, Qwen3, Qwen3-Next). <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [MoE Hardware Configuration card.yaml](skills/nemo-mbridge-perf-moe-hardware-configs/card.yaml) <br>
- [Performance Tuning Guide](docs/performance-guide.md) <br>
- [Megatron Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks and tables] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 positive skill-activation task in a k8s-sandbox environment with 1 attempt per task and a 50% pass threshold. <br>

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



## Evaluation Results: <br>
| Dimension | Num | Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) | Codex (`openai/openai/gpt-5.5`) |
|---|---:|---:|---:|
| Security | 1 | 100% (+0%) | 100% (+0%) |
| Correctness | 1 | 100% (+100%) | 100% (+0%) |
| Discoverability | 1 | 100% (+50%) | 94% (+44%) |
| Effectiveness | 1 | 100% (+79%) | 100% (+7%) |
| Efficiency | 1 | 100% (+70%) | 100% (+88%) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
3990c04c (source: git SHA, committed 2026-07-26) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
