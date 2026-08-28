## Description: <br>
Operational guide for sharding a VLM vision encoder across the language model's context-parallel ranks in Megatron-Bridge, including config knobs, code anchors, load-balance pitfalls, and measured impact. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers optimizing VLM training performance who need to reduce peak memory and improve throughput by sharding the vision encoder's image inputs across context-parallel ranks in Megatron-Bridge. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Megatron Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) <br>
- [SKILL.md](SKILL.md) <br>
- [card.yaml](card.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 evaluation tasks (2 positive, 1 negative) in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the skill produces correct answers against the reference. <br>
- Discoverability: Whether the right skill is loaded and activated when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and expected workflow (goal completion + behavior adherence). <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage (routing quality and productive tool use). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 67% → 100% (+32 points) | 59% → 95% (+36 points) |
| Security | 100% → 100% (±0 points) | 67% → 100% (+33 points) |
| Correctness | 47% → 100% (+53 points) | 60% → 93% (+33 points) |
| Discoverability | 83% → 100% (+17 points) | 58% → 96% (+38 points) |
| Effectiveness | 34% → 99% (+65 points) | 50% → 88% (+38 points) |
| Efficiency | 73% → 100% (+27 points) | 62% → 100% (+38 points) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
1.0.0+b7643bd (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
